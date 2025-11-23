import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect, session, send_file
from PIL import Image
import numpy as np
import cv2
import pywt
import io
from flask import Flask, render_template

# PDF imports
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# utils
# OLD → from backend.utils.plotly_3d import build_plotly_3d
# ---- NOW IMPORT UTILS CORRECTLY ----
from utils.plotly_3d import build_plotly_3d_with_bscan
from utils.dr_inference import run_dr_unet
from utils.glaucoma_inference import run_glaucoma_unet
from utils.cataract_3d import build_cataract_heightmap


warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------
# ResNet50 — 6-channel classifier
# ------------------------------------------------
class ResNet50_6ch(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        try:
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            )
        except Exception:
            resnet = models.resnet50(pretrained=pretrained)

        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            6, old_conv.out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            resnet.conv1.weight[:, :3] = old_conv.weight
            mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
            resnet.conv1.weight[:, 3:] = mean_rgb

        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)

# ------------------------------------------------
# Flask setup
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "replace_this_with_a_secure_random_key"

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MASKS_FOLDER = os.path.join(STATIC_DIR, "masks")
os.makedirs(MASKS_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------
# Load classifier
# ------------------------------------------------
CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]
CLASS_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_wavelet_resnet50.pt")

classifier = ResNet50_6ch(num_classes=len(CLASS_NAMES))

if os.path.exists(CLASS_MODEL_PATH):
    try:
        checkpoint = torch.load(CLASS_MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and ("model" in checkpoint or "state_dict" in checkpoint):
            state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint

        fixed_state = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("model.", "")
            if not k.startswith("resnet."):
                k = f"resnet.{k}"
            fixed_state[k] = v

        classifier.load_state_dict(fixed_state, strict=False)
        print("✅ Classifier loaded")
    except Exception as e:
        print("❌ Failed to load classifier:", e)
else:
    print("⚠️ Classifier weights not found!")

classifier = classifier.to(DEVICE)
classifier.eval()

# ------------------------------------------------
# FUNDUS Detector
# ------------------------------------------------
class FundusDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)

fundus_detector = FundusDetector().to(DEVICE)
FUNDUS_MODEL_PATH = os.path.join(BASE_DIR, "models", "fundus_vs_nonfundus.pt")

if os.path.exists(FUNDUS_MODEL_PATH):
    try:
        fd_state = torch.load(FUNDUS_MODEL_PATH, map_location=DEVICE)
        new_state = {}
        for k, v in fd_state.items():
            if k.startswith("features") or k.startswith("classifier"):
                new_state[f"model.{k}"] = v
            else:
                new_state[k] = v
        fundus_detector.load_state_dict(new_state, strict=False)
    except:
        pass

fundus_detector.eval()

# ------------------------------------------------
# Preprocess Wavelet
# ------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def preprocess_wavelet(img_pil, wavelet="db2", size=(224,224)):
    img = img_pil.convert("RGB").resize(size)
    img_np = np.array(img).astype(np.float32) / 255.0

    green = img_np[:,:,1]
    try:
        _, (cH,cV,cD) = pywt.dwt2(green, wavelet)
    except:
        cH = cv2.Sobel(green, cv2.CV_32F,1,0)
        cV = cv2.Sobel(green, cv2.CV_32F,0,1)
        cD = np.zeros_like(green)

    resize = lambda x: cv2.resize(x,(img_np.shape[1],img_np.shape[0]))
    wave = np.stack([resize(cH), resize(cV), resize(cD)], axis=-1)
    wave = (wave - wave.min()) / (wave.max()-wave.min()+1e-8)

    rgb  = torch.tensor(img_np).permute(2,0,1)
    wave = torch.tensor(wave).permute(2,0,1)

    x = torch.cat([rgb, wave], dim=0)
    x[:3] = (x[:3] - IMAGENET_MEAN)/IMAGENET_STD

    return x.unsqueeze(0)

# ------------------------------------------------
# MAIN ROUTES
# ------------------------------------------------
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

# NEW RESULT PAGE
@app.route("/result", methods=["GET"])
def result_page():
    return render_template(
        "result.html",
        disease=session.get("disease"),
        original_image=session.get("original_image"),
        result_image=session.get("result_image"),
        bbox_count=session.get("bbox_count"),
        cls_acc=session.get("cls_acc"),
        dice_score=session.get("dice_score"),
        iou_score=session.get("iou_score"),
        seg_acc=session.get("seg_acc"),
        severity=session.get("severity"),
        cdr=session.get("cdr"),
        plot3d=session.get("plot3d"),
        probs=session.get("probs")
    )

# 3D page
@app.route("/plot3d/<html_file>")
def plot3d(html_file):
    return render_template(
        "plotly3d.html",
        html_file=html_file,
        back_url="/result"
    )

# ------------------------------------------------
# PREDICT
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # === LOAD IMAGE ===
    if "image" not in request.files:
        return redirect("/")
    file = request.files["image"]
    if file.filename == "":
        return redirect("/")

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    img_pil = Image.open(filepath).convert("RGB")

    # === FUNDUS DETECTION ===
    prep_fd = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img_fd = prep_fd(img_pil).unsqueeze(0).to(DEVICE)

    is_fundus = True
    if os.path.exists(FUNDUS_MODEL_PATH):
        with torch.no_grad():
            out = torch.softmax(fundus_detector(img_fd),1)[0]
            if out[0] > 0.6:
                is_fundus=False

    if not is_fundus:
        session.clear()
        session["disease"]="Invalid Image"
        session["original_image"]=file.filename
        return redirect("/result")

    # === CLASSIFICATION ===
    x = preprocess_wavelet(img_pil).to(DEVICE)
    with torch.no_grad():
        out = classifier(x)
        prob = torch.softmax(out,1)[0].cpu().numpy()

    probs_list = [float(p) for p in prob]
    cls_acc = float(np.max(prob)*100)
    disease = CLASS_NAMES[int(np.argmax(prob))]

    # === ALWAYS PRINT CONFIDENCE SCORES IN TERMINAL ===
    print("\n🔍 Classification Probabilities:")
    for cname, p in zip(CLASS_NAMES, probs_list):
        print(f"  {cname}: {p*100:.2f}%")
    print("Predicted:", disease, "| Confidence:", f"{cls_acc:.2f}%")
    print(f"➡ Proceeding with {disease} pipeline...\n")

    # Default values
    result_filename = None
    bbox_count = 0
    cdr = None
    dice_score = None
    iou_score = None
    seg_acc = None
    severity = None

    # -------------------------
    # DIABETIC RETINOPATHY
    # -------------------------
    if disease == "Diabetic Retinopathy":
        res = run_dr_unet(filepath, UPLOAD_FOLDER)
        if isinstance(res,(tuple,list)):
            result_filename = res[0]
            bbox_count = res[1] if len(res)>1 else 0
        else:
            result_filename=res

        if bbox_count>=50: severity="Severe"
        elif bbox_count>=15: severity="Moderate"
        elif bbox_count>0: severity="Mild"
        else: severity="None"

        # OLD → plot3d_file = build_plotly_3d(filepath,UPLOAD_FOLDER)
        files = build_plotly_3d_with_bscan(filepath, UPLOAD_FOLDER)   # ✅ FIX
        plot3d_file = files["html_file"]

        session.update({
            "disease":disease,
            "cls_acc":round(cls_acc,2),
            "dice_score":None,
            "iou_score":None,
            "seg_acc":None,
            "severity":severity,
            "cdr":None,
            "bbox_count":bbox_count,
            "original_image":file.filename,
            "result_image":result_filename,
            "probs":probs_list,
            "plot3d":plot3d_file
        })

        return redirect("/result")

    # -------------------------
    # GLAUCOMA
    # -------------------------
    if disease=="Glaucoma":
        res = run_glaucoma_unet(filepath,UPLOAD_FOLDER)
        try:
            cdr=float(res[0])
            result_filename=res[1]
        except:
            pass

        severity = "High" if cdr and cdr>=0.6 else "Low"

        # OLD → plot3d_file = build_plotly_3d(filepath,UPLOAD_FOLDER)
        files = build_plotly_3d_with_bscan(filepath, UPLOAD_FOLDER)     # ✅ FIX
        plot3d_file = files["html_file"]

        session.update({
            "disease":disease,
            "cls_acc":round(cls_acc,2),
            "dice_score":None,
            "iou_score":None,
            "seg_acc":None,
            "severity":severity,
            "cdr":round(cdr,2),
            "bbox_count":0,
            "original_image":file.filename,
            "result_image":result_filename,
            "probs":probs_list,
            "plot3d":plot3d_file
        })

        return redirect("/result")

    # -------------------------
    # CATARACT
    # -------------------------
    if disease=="Cataract":
        severity="Detected"

        # OLD → plot3d_file = build_plotly_3d(filepath,UPLOAD_FOLDER)
        files = build_plotly_3d_with_bscan(filepath, UPLOAD_FOLDER)     # ✅ FIX
        plot3d_file = files["html_file"]

        session.update({
            "disease":disease,
            "cls_acc":round(cls_acc,2),
            "dice_score":None,
            "iou_score":None,
            "seg_acc":None,
            "severity":severity,
            "cdr":None,
            "bbox_count":0,
            "original_image":file.filename,
            "result_image":None,
            "probs":probs_list,
            "plot3d":plot3d_file
        })

        return redirect("/result")

    # -------------------------
    # NORMAL
    # -------------------------
    # OLD → plot3d_file = build_plotly_3d(filepath,UPLOAD_FOLDER)
    files = build_plotly_3d_with_bscan(filepath, UPLOAD_FOLDER)     # ✅ FIX
    plot3d_file = files["html_file"]

    session.update({
        "disease":disease,
        "cls_acc":round(cls_acc,2),
        "dice_score":None,
        "iou_score":None,
        "seg_acc":None,
        "severity":None,
        "cdr":None,
        "bbox_count":0,
        "original_image":file.filename,
        "result_image":None,
        "probs":probs_list,
        "plot3d":plot3d_file
    })

    return redirect("/result")

# ------------------------------------------------
# PDF DOWNLOAD
# ------------------------------------------------
@app.route("/download_report")
def download_report():

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    page_w, page_h = A4

    y = page_h - 50
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, y, "Eye Disease Analysis Report")
    y -= 40
    pdf.setFont("Helvetica", 12)

    def L(text):
        nonlocal y
        pdf.drawString(50, y, str(text))
        y -= 18

    disease=session.get("disease")
    cls_acc=session.get("cls_acc")
    severity=session.get("severity")
    cdr=session.get("cdr")
    bbox=session.get("bbox_count")
    orig=session.get("original_image")
    resimg=session.get("result_image")

    L(f"Disease: {disease}")
    L(f"Confidence: {cls_acc}%")
    L(f"Severity: {severity}")

    if disease=="Diabetic Retinopathy":
        L(f"Exudates: {bbox}")
    if disease=="Glaucoma":
        L(f"CDR Ratio: {cdr}")

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        download_name="Analysis_Report.pdf",
        as_attachment=True,
        mimetype="application/pdf"
    )

# ------------------------------------------------
# RUN APP
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
