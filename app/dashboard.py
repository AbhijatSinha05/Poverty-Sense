import gradio as gr
import os
from PIL import Image

HEATMAP_PATH = "poverty_heatmap.png"
REPORT_DIR = "reports"

# Load heatmap
heatmap_img = Image.open(HEATMAP_PATH)

# Get available districts from reports folder
district_files = sorted(os.listdir(REPORT_DIR))
district_names = [f.replace("_report.txt","").title() for f in district_files]

def load_report(district_name):
    filename = district_name.lower() + "_report.txt"
    path = os.path.join(REPORT_DIR, filename)
    with open(path, "r") as f:
        return f.read()

# Gradio UI
with gr.Blocks(title="PovertySense Dashboard") as demo:
    gr.Markdown("# 🌍 PovertySense – District Poverty Mapping Dashboard")

    gr.Markdown("### Tamil Nadu Poverty Heatmap (ViT + GNN)")
    gr.Image(heatmap_img, label="District Poverty Heatmap")

    gr.Markdown("### District-Level Area Report")
    dropdown = gr.Dropdown(district_names, label="Select District")
    report_box = gr.Textbox(label="Generated Area Report", lines=12)

    dropdown.change(fn=load_report, inputs=dropdown, outputs=report_box)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
