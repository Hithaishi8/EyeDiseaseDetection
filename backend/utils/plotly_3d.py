import cv2
import numpy as np
import plotly.graph_objects as go
import os

def build_plotly_3d_with_bscan(image_path, save_folder):
    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- BASIC PIPELINE ----
    # Smooth (denoise)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Normalize 0–1
    gray_norm = cv2.normalize(gray_blur.astype("float32"), None, 0, 1, cv2.NORM_MINMAX)

    # Depth map (Pseudo-OCT)
    Z = gray_norm * 50  # scale depth

    # Meshgrid for 3D Plot
    h, w = gray.shape
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    X, Y = np.meshgrid(x, y)

    # ---- MIDDLE B-SCAN EXTRACTION ----
    mid_x = w // 2  # vertical slice (middle)
    mid_y = h // 2  # horizontal slice (middle)

    bscan_vertical = Z[:, mid_x]          # (height, )
    bscan_horizontal = Z[mid_y, :]        # (width, )

    # Save B-scan images
    os.makedirs(save_folder, exist_ok=True)

    # Vertical B-scan image (reshape to 2D)
    bscan_v_img = cv2.normalize(bscan_vertical.reshape(h, 1), None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(save_folder, "bscan_vertical.png"), bscan_v_img)

    # Horizontal B-scan
    bscan_h_img = cv2.normalize(bscan_horizontal.reshape(1, w), None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(save_folder, "bscan_horizontal.png"), bscan_h_img)

    # ---- BUILD INTERACTIVE 3D PLOT ----
    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=Z,
        x=X,
        y=Y,
        colorscale="Inferno",
        opacity=0.95
    ))

    fig.update_layout(
        title="3D Retina Surface + Middle B-Scan",
        scene=dict(
            xaxis_title='Width',
            yaxis_title='Height',
            zaxis_title='Depth',
            zaxis=dict(range=[0, 50])
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Save 3D plot
    html_name = "plotly_3d_view.html"
    html_path = os.path.join(save_folder, html_name)
    fig.write_html(html_path)

    return {
        "html_file": html_name,
        "bscan_vertical": "bscan_vertical.png",
        "bscan_horizontal": "bscan_horizontal.png"
    }
