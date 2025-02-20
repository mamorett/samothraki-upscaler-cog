from cog import BasePredictor, Input, Path
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, LCMScheduler
from diffusers.models import AutoencoderKL
from cog import BasePredictor, Input, Path
import cv2
import pywt
import random
import os
from torchvision import transforms


class UpscalerModel(torch.nn.Module):
    def __init__(self, scale=2.0):  # Add self parameter and default value
        super(UpscalerModel, self).__init__()  # Call parent class init first
        self.scale = scale
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move model to device once during initialization

    def forward(self, x):
        # Define the forward pass of your model
        return x

    def predict(self, image, upscale_model_path):
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Define transformation (resize the image for processing)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((int(image.height * self.scale), int(image.width * self.scale))),
            transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
        ])

        try:
            image_tensor = transform(image).to(self.device)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to transform image: {e}")

        # Load the state dictionary from the checkpoint
        state_dict = torch.load(upscale_model_path)

        # Optionally, remove a prefix from keys if needed
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")
            new_state_dict[new_key] = value

        try:
            self.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")

        self.eval()

        with torch.no_grad():
            upscaled_tensor = self(image_tensor)

        upscaled_image = upscaled_tensor.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255
        upscaled_image = Image.fromarray(upscaled_image.astype(np.uint8))
        return upscaled_image

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define paths relative to /src directory
        base_path = "/src"
        controlnet_path_1 = os.path.join(base_path, "controlnet-cache", "control_v11f1e_sd15_tile.safetensors")  # Update filename
        controlnet_path_2 = os.path.join(base_path, "controlnet-cache", "control_v11p_sd15_inpaint_fp16.safetensors")  # Update filename
        sd_model_path = os.path.join(base_path, "model-cache", "juggernaut_reborn.safetensors")  # Update filename
        vae_path = os.path.join(base_path, "model-cache", "vae-ft-mse-840000-ema-pruned.ckpt")  # Update filename
        lora_weights1_path = os.path.join(base_path, "loras-cache", "lcm-lora-sdv1-5.safetensors")  # Update filename
        lora_weights2_path = os.path.join(base_path, "loras-cache", "more_details.safetensors")  # Update filename

        # Define upscaler model paths
        self.upscalers = {
            "4x_NMKD-Siax_200k": os.path.join(base_path, "upscaler-cache", "4x_NMKD-Siax_200k.pth"),
            "4xSSDIRDAT": os.path.join(base_path, "upscaler-cache", "4xSSDIRDAT.pth")
        }

        # Load ControlNet models
        controlnets = [
            ControlNetModel.from_single_file(
                controlnet_path_1,
                torch_dtype=torch.float16,
            ),
            ControlNetModel.from_single_file(
                controlnet_path_2,
                torch_dtype=torch.float16,
            ),
        ]
        
        # Load main pipeline
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
            sd_model_path,
            controlnet=controlnets,
            torch_dtype=torch.float16,
            use_safetensors=True,
            # safety_checker=None,
            control_guidance_end=[0.5, 1.0]
        )
        self.pipe.enable_model_cpu_offload()
        # Load and set VAE
        vae = AutoencoderKL.from_single_file(
            vae_path,
            torch_dtype=torch.float16
        )
        self.pipe.vae = vae
        
        # Load LoRA weights
        self.pipe.load_lora_weights(lora_weights1_path)
        self.pipe.fuse_lora(lora_scale=1.0)
        self.pipe.load_lora_weights(lora_weights2_path)
        self.pipe.fuse_lora(lora_scale=0.25)
        
        # Set scheduler and enable FreeU
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        self.pipe.to(self.device)

    def predict(
        self,
        image: Path = Input(description="Input image to process"),
        upscaler: str = Input(description="Upscaler", default="4x_NMKD-Siax_200k", choices=["4x_NMKD-Siax_200k", "4xSSDIRDAT"]),
        upscale_by: float = Input(description="Upscale By", default=2.0),        
        num_inference_steps: int = Input(description="Number of inference steps", default=20),
        denoise: float = Input(description="Denoise, use 1.0 for best results", default=1.0),
        hdr: float = Input(description="HDR effect intensity", default=0.0),
        guidance_scale: float = Input(description="Guidance scale", default=3.0),
        color_correction: bool = Input(description="GWavelet Color Correction", default=True),

    ) -> Path:
        """Run a single prediction on the model"""
        input_image = Image.open(image)

        # Apply upscaling with selected model and scale factor
        # Create upscaler model with selected scale factor
        upscaler_model = UpscalerModel(upscale_by)
        
        try:
            # Get the path for the selected upscaler model
            upscaler_path = self.upscalers[upscaler]
            # Upscale the image
            upscaled_image = upscaler_model.predict(input_image, upscaler_path)
        except Exception as e:
            raise RuntimeError(f"Error during upscaling: {e}")

        torch.cuda.empty_cache()

        if hdr > 0.1:
            condition_image = self.create_hdr_effect(upscaled_image, hdr)
        else:
            condition_image = upscaled_image
        
        # Process the image
        processed_image = self.process_image(
            condition_image,
            num_inference_steps,
            denoise,
            guidance_scale
        )
        
        if color_correction:
            # Apply wavelet color transfer
            condition_image_numpy = np.array(condition_image)
            processed_image_numpy = np.array(processed_image)  # Convert to numpy array
            final_result_numpy = self.wavelet_color_transfer(condition_image_numpy, processed_image_numpy)
            final_result = Image.fromarray(final_result_numpy)  # Convert back to PIL Image
        else:
            final_result = processed_image

        # Save and return the result
        output_path = Path("/tmp/output.png")
        final_result.save(output_path)
        return output_path

    def create_hdr_effect(self, original_image, hdr):
        if hdr == 0:
            return original_image
        cv_original = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        factors = [1.0 - 0.9 * hdr, 1.0 - 0.7 * hdr, 1.0 - 0.45 * hdr,
                  1.0 - 0.25 * hdr, 1.0, 1.0 + 0.2 * hdr,
                  1.0 + 0.4 * hdr, 1.0 + 0.6 * hdr, 1.0 + 0.8 * hdr]
        images = [cv2.convertScaleAbs(cv_original, alpha=factor) for factor in factors]
        merge_mertens = cv2.createMergeMertens()
        hdr_image = merge_mertens.process(images)
        hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
        return Image.fromarray(cv2.cvtColor(hdr_image_8bit, cv2.COLOR_BGR2RGB))

    def wavelet_color_transfer(self, img1, img2, wavelet='haar', level=2):
        img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
        
        l1, a1, b1 = cv2.split(img1_lab)
        l2, a2, b2 = cv2.split(img2_lab)
        
        def wavelet_transfer(channel1, channel2):
            coeffs1 = pywt.wavedec2(channel1, wavelet, level=level)
            coeffs2 = pywt.wavedec2(channel2, wavelet, level=level)
            coeffs2_transferred = list(coeffs2)
            coeffs2_transferred[0] = coeffs1[0]
            return pywt.waverec2(coeffs2_transferred, wavelet)
        
        a2_corrected = np.clip(wavelet_transfer(a1, a2), 0, 255).astype(np.uint8)
        b2_corrected = np.clip(wavelet_transfer(b1, b2), 0, 255).astype(np.uint8)
        
        corrected_lab = cv2.merge((l2, a2_corrected, b2_corrected))
        return cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)

    def process_tile(self, tile, num_inference_steps, strength, guidance_scale):
        prompt = "masterpiece, best quality, highres"
        negative_prompt = "low quality, normal quality, ugly, blurry, blur, lowres, bad anatomy, bad hands, cropped, worst quality"
        
        tile_list = [tile] * 2
        
        options = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": tile,
            "control_image": tile_list,
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": [1.0, 0.55],
            "generator": torch.Generator(device=self.device).manual_seed(random.randint(0, 2147483647)),
        }
        
        return np.array(self.pipe(**options).images[0])
    

    def create_gaussian_weight(self, tile_size, sigma=0.3):
        x = np.linspace(-1, 1, tile_size)
        y = np.linspace(-1, 1, tile_size)
        xx, yy = np.meshgrid(x, y)
        gaussian_weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        return gaussian_weight

    def process_image(self, condition_image, num_inference_steps, strength, guidance_scale):
        # Prepare the condition image       
        W, H = condition_image.size
        
        # Set up tiling
        tile_size = 512
        overlap = 64
        
        # Calculate number of tiles
        num_tiles_x = -(-W // (tile_size - overlap))  # Ceiling division
        num_tiles_y = -(-H // (tile_size - overlap))
        
        # Create result canvas
        result = np.zeros((H, W, 3), dtype=np.float32)
        weight_sum = np.zeros((H, W, 1), dtype=np.float32)
        
        # Create gaussian weight
        gaussian_weight = self.create_gaussian_weight(tile_size)
        
        # Process tiles
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                top = i * (tile_size - overlap)
                left = j * (tile_size - overlap)
                bottom = min(top + tile_size, H)
                right = min(left + tile_size, W)
                
                # Extract and process tile
                tile = condition_image.crop((left, top, right, bottom))
                tile = tile.resize((tile_size, tile_size))
                processed_tile = self.process_tile(tile, num_inference_steps, strength, guidance_scale)
                
                # Resize processed tile if needed
                if (bottom - top, right - left) != (tile_size, tile_size):
                    processed_tile = cv2.resize(processed_tile, (right - left, bottom - top))
                    tile_weight = cv2.resize(gaussian_weight, (right - left, bottom - top))
                else:
                    tile_weight = gaussian_weight
                
                # Add to result with weighting
                result[top:bottom, left:right] += processed_tile * tile_weight[:, :, np.newaxis]
                weight_sum[top:bottom, left:right] += tile_weight[:, :, np.newaxis]
        
        # Normalize result
        final_result = (result / weight_sum).astype(np.uint8)        
        return Image.fromarray(final_result)