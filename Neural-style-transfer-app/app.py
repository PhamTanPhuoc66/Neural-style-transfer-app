# --- IMPORT NECESSARY LIBRARIES ---
import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# --- IMPORT THE MODEL CLASS FROM YOUR OTHER FILE ---

try:
    # Updated to match the filename of the fixed model
    from model import StyleTransferModel, tensor_to_image
except ImportError:
    print("Error: Could not import StyleTransferModel.")
    print("Please make sure the file 'style_transfer_model.py' exists in the same directory.")
    exit()

# --- SETUP AND CONFIGURATION ---
PICTURES_DIR = "pictures"
if not os.path.exists(PICTURES_DIR):
    os.makedirs(PICTURES_DIR)
    print(f"Created directory: {PICTURES_DIR}")
    print("Please add some style images (e.g., 'style1.jpg', 'style2.jpg') to this directory.")


example_styles = [os.path.join(PICTURES_DIR, f) for f in os.listdir(PICTURES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Determine the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UI HELPER FUNCTIONS ---
def start_processing():
    """Disables the button and changes its text to show a processing state."""
    return gr.Button(value="Đang xử lý... 🎨", interactive=False)

def end_processing():
    """Resets the button to its original state after processing is complete."""
    return gr.Button(value="Tạo Ảnh! (Stylize!)", interactive=True)

# --- IMAGE PROCESSING FUNCTION ---
def stylize_image(content_img, style_img, style_w, content_w, epochs, learning_rate, init_noise):
    """
    The main processing function that takes inputs from the Gradio interface,
    runs the style transfer model, and returns the output image.
    """
    # Check for valid inputs
    if content_img is None:
        raise gr.Error("Vui lòng tải lên hoặc chọn một Ảnh Nội Dung (Content Image).")
    if style_img is None:
        raise gr.Error("Vui lòng tải lên hoặc chọn một Ảnh Phong Cách (Style Image).")

    # --- Pre-process images ---
    # Get the original size of the content image
    w, h = content_img.size

    # To prevent memory errors, cap the maximum dimension of the image
    max_dim = 512 if torch.cuda.is_available() else 256
    if max(w, h) > max_dim:
        # Scale down while preserving aspect ratio
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        image_size = (new_h, new_w)
        print(f"Image was too large, resized from {(w,h)} to {image_size}")
    else:
        # transforms.Resize expects (height, width)
        image_size = (h, w)

    # Define image transformations with the dynamic size
    loader = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # Convert PIL images from Gradio to tensors
    content_tensor = loader(content_img).unsqueeze(0).to(device, torch.float)
    style_tensor = loader(style_img).unsqueeze(0).to(device, torch.float)

    # --- Initialize and run the model ---
    # Instantiate the model with parameters from the Gradio sliders
    style_transfer = StyleTransferModel(
        style_weight=style_w,
        content_weight=content_w
    )

    # Run the style transfer process, now including the learning rate
    output_tensor = style_transfer.run(
        content_img_tensor=content_tensor,
        style_img_tensor=style_tensor,
        epochs=int(epochs),
        lr=learning_rate, # Pass the learning rate from the UI
        init_noise=init_noise
    )

    # --- Post-process the output ---
    # Convert the output tensor back to a PIL image for display
    final_image = tensor_to_image(output_tensor)
    
    return final_image


# --- GRADIO INTERFACE DEFINITION ---
with gr.Blocks(theme=gr.themes.Soft(), title="Neural Style Transfer") as demo:
    gr.Markdown("# 🎨 Neural Style Transfer")
    gr.Markdown("Tải ảnh lên, chọn phong cách, điều chỉnh tham số và xem điều kỳ diệu!")

    with gr.Row(variant='panel'):
        # --- CỘT BÊN TRÁI: INPUT ---
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("## 1. Đầu vào")
            content_image = gr.Image(type="pil", label="Ảnh Nội Dung (Content)", height=300)
            style_image = gr.Image(type="pil", label="Ảnh Phong Cách (Style)", height=300)
            
            if example_styles:
                gr.Examples(examples=example_styles, inputs=style_image, label="Hoặc chọn phong cách có sẵn")

        # --- CỘT BÊN PHẢI: OUTPUT VÀ THAM SỐ ---
        with gr.Column(scale=2):
            gr.Markdown("## 2. Kết quả & Tinh chỉnh")
            
            # 1. Ảnh kết quả
            output_image = gr.Image(label="Ảnh Kết Quả", height=400, interactive=False)
            
            # 2. Nút chạy
            run_button = gr.Button("Tạo Ảnh ✨", variant="primary")
            
            # 3. Các tham số (hiển thị sẵn)
            style_weight = gr.Slider(label="Độ ưu tiên Phong cách", minimum=10.0, maximum=1000.0, value=50.0, step=10.0)
            content_weight = gr.Slider(label="Độ ưu tiên Nội dung", minimum=1, maximum=100, value=1, step=1)
            epochs = gr.Slider(label="Số bước tối ưu (Epochs)", minimum=100, maximum=10000, value=500, step=100)
            learning_rate = gr.Slider(label="Tốc độ học (Learning Rate)", minimum=0.01, maximum=0.51, value=0.1, step=0.05)
            init_noise = gr.Checkbox(label="Khởi tạo từ White Noise", value=False)

    # --- KẾT NỐI SỰ KIỆN ---
    run_button.click(
        fn=start_processing,
        inputs=None,
        outputs=run_button
    ).then(
        fn=stylize_image,
        inputs=[content_image, style_image, style_weight, content_weight, epochs, learning_rate, init_noise],
        outputs=output_image,
        api_name="stylize"
    ).then(
        fn=end_processing,
        inputs=None,
        outputs=run_button
    )

if __name__ == "__main__":
    demo.launch(share=True)
