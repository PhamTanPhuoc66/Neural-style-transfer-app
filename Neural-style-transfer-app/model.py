# --- IMPORT NECESSARY LIBRARIES ---
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import os

class StyleTransferModel:
    """
    A class that encapsulates the entire Neural Style Transfer pipeline.
    
    This class holds the VGG model, loss calculation functions, and the
    optimization loop to generate the final stylized image. It is designed
    for easy reuse and integration into a larger application.
    """
    def __init__(self, content_layer_n=21, 
                 style_layers_indices=[0, 5, 10, 19, 28],
                 style_weight=1e6, content_weight=1):
        """
        Initializes the model and its parameters.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.content_layer = content_layer_n
        self.style_layers = style_layers_indices
        self.style_weight = style_weight
        self.content_weight = content_weight
        
        self.feature_extractor = self._get_feature_extractor().to(self.device).eval()
        print(f"Model loaded and running on: {self.device}")

    def _get_feature_extractor(self):
        """
        Internal method to build a sub-model from VGG19.
        """
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        for param in vgg19.parameters():
            param.requires_grad_(False)
            
        max_layer_index = max(self.content_layer, *self.style_layers)
        model = nn.Sequential()
        
        for i in range(max_layer_index + 1):
            model.add_module(str(i), vgg19[i])
            
        return model

    def _extract_features(self, x):
        """
        Extracts content and style feature maps from an input image.
        """
        content_features = []
        style_features = []
        
        for name, layer in self.feature_extractor.named_children():
            x = layer(x)
            layer_index = int(name)
            
            if layer_index == self.content_layer:
                content_features.append(x)
                
            if layer_index in self.style_layers:
                style_features.append(x)
        
        return content_features, style_features
    def _gram_matrix(self, input_tensor):
        """
        Calculates the Gram matrix of a feature map.
        """
        b, c, h, w = input_tensor.size()
        features = input_tensor.view(b * c, h * w)
        G = torch.mm(features, features.t())

        # Chuẩn hóa ma trận Gram bằng số phần tử của chính nó (c*c).
        # Đây là một phương pháp ổn định hơn về mặt số học.
        return G.div(c * c)

    def run(self, content_img_tensor, style_img_tensor, epochs=300, lr=0.01, init_noise=False):
        """
        The main function to run the style transfer algorithm.
        """
        print("Starting the style transfer process...")

        # --- Trích xuất đặc trưng của ảnh mục tiêu ---
        with torch.no_grad():
            target_content_features, _ = self._extract_features(content_img_tensor.to(torch.float32))
            _, target_style_features = self._extract_features(style_img_tensor.to(torch.float32))
            target_style_grams = [self._gram_matrix(f) for f in target_style_features]

        # --- Khởi tạo ảnh ---
        if init_noise:
            generated_image = torch.randn_like(content_img_tensor).to(self.device).requires_grad_(True)
        else:
            generated_image = content_img_tensor.clone().requires_grad_(True)

        # --- Cài đặt Optimizer ---
        optimizer = optim.Adam([generated_image], lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
        use_amp = self.device.type == 'cuda'
        scaler = torch.amp.GradScaler(enabled=use_amp)

        print("Starting optimization loop...")
        for epoch in range(epochs):
            optimizer.zero_grad()

            # --- Tính toán loss ở float32 để đảm bảo ổn định ---
            gen_content_features, gen_style_features = self._extract_features(generated_image)

            content_loss_raw = nn.functional.mse_loss(gen_content_features[0].to(torch.float32), target_content_features[0])

            style_loss_raw = 0
            for i in range(len(target_style_grams)):
                gen_gram = self._gram_matrix(gen_style_features[i].to(torch.float32))
                style_loss_raw += nn.functional.mse_loss(gen_gram, target_style_grams[i])

            content_loss = self.content_weight * content_loss_raw
            style_loss = self.style_weight * style_loss_raw
            total_loss = content_loss + style_loss

            # --- Backprop và Gradient Clipping ---
            scaler.scale(total_loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)  # Unscale gradients 
            torch.nn.utils.clip_grad_norm_([generated_image], max_norm=1.0) # Đặt giới hạn cho độ lớn của gradient
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                generated_image.clamp_(0, 1)

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs} | "
                    f"Total Loss: {total_loss.item():.4f} | "
                    f"Content Loss: {content_loss.item():.4f} | "
                    f"Style Loss: {style_loss.item():.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}")

        print("Optimization finished!")
        return generated_image

# --- UTILITY FUNCTIONS  ---

def image_loader(image_source, loader, device):
    """Loads an image from a URL or local path, resizes it, and converts it to a tensor."""
    if isinstance(image_source, str) and image_source.startswith('http'):
        try:
            response = requests.get(image_source)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except requests.exceptions.RequestException as e:
            raise IOError(f"Could not download image from URL: {e}")
    else:
        if not os.path.exists(image_source):
             raise FileNotFoundError(f"Image file not found at: {image_source}")
        image = Image.open(image_source).convert('RGB')
    
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def tensor_to_image(tensor):
    """Converts a tensor back to a PIL image for viewing or saving."""
    image = tensor.cpu().clone().squeeze(0)
    unloader = transforms.ToPILImage()
    image = unloader(image)
    return image

