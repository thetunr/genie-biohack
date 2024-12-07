import torch
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
from PIL import Image
from skimage import io
import skimage

def ycbcr(image):
    """
    Applies a YCbCr mask to an input image and 
    saves the resultant image

    Inputs:
        image - (str) RBG image file path

    Outputs:
        mask - (str) YCbCr image file path
    """

    # Read RGB image
    RGB = io.imread(image)

    # Subset dimensions
    R = RGB[:, :, 0]
    G = RGB[:, :, 1]
    B = RGB[:, :, 2]

    # Reduce to skin range
    R = np.where(R < 95, 0, R)
    G = np.where(G < 40, 0, G)
    B = np.where(B < 20, 0, B)

    R = np.where(R < G, 0, R)
    R = np.where(R < B, 0, R)
    R = np.where(abs(R - G) < 15, 0, R)

    R = np.where(G == 0, 0, R)
    R = np.where(B == 0, 0, R)

    B = np.where(R == 0, 0, B)
    B = np.where(G == 0, 0, B)

    G = np.where(R == 0, 0, G)
    G = np.where(B == 0, 0, G)

    # Stack into RGB
    RGB = np.stack([R, G, B], axis = 2)

    # Convert to YCBCR color-space
    YCBCR = color.rgb2ycbcr(RGB)

    # Subset dimensions
    Y = YCBCR[:, :, 0]
    Cb = YCBCR[:, :, 1]
    Cr = YCBCR[:, :, 2]

    # Subset to skin range
    Y = np.where(Y < 80, 0, Y)
    Cb = np.where(Cb < 85, 0, Cb)
    Cr = np.where(Cr < 135, 0, Cr)

    Cr = np.where(Cr >= (1.5862*Cb) + 20, 0, Cr)
    Cr = np.where(Cr <= (0.3448*Cb) + 76.2069, 0, Cr)
    Cr = np.where(Cr <= (-4.5652*Cb) + 234.5652, 0, Cr)
    Cr = np.where(Cr >= (-1.15*Cb) + 301.75, 0, Cr)
    Cr = np.where(Cr >= (-2.2857*Cb) + 432.85, 0, Cr)

    Y = np.where(Cb == 0, 0, Y)
    Y = np.where(Cr == 0, 0, Y)

    Cb = np.where(Y == 0, 0, Cb)
    Cb = np.where(Cr == 0, 0, Cb)

    Cr = np.where(Y == 0, 0, Cr)
    Cr = np.where(Cb == 0, 0, Cr)

    # Stack into skin region
    skinRegion = np.stack([Y, Cb, Cr], axis = 2)
    skinRegion = np.where(skinRegion != 0, 255, 0)
    skinRegion = skinRegion.astype(dtype = "uint8")

    # Apply mask to original RGB image
    mask = np.array(RGB)
    mask = np.where(skinRegion != 0, mask, 0)

    new_filepath = 'ycbcr/{}'.format(image)
    Image.save(new_filepath)

    return new_filepath

def ITA(image):
    """
    Calculates the individual typology angle (ITA) for a given 
    RGB image.

    Inputs:
        image - (str) RGB image file path

    Outputs:
        ITA - (float) individual typology angle
    """

    # Convert to CIE-LAB color space
    RGB = Image.open(image)
    CIELAB = np.array(color.rgb2lab(RGB))

    # Get L and B (subset to +- 1 std from mean)
    L = CIELAB[:, :, 0]
    L = np.where(L != 0, L, np.nan)
    std, mean = np.nanstd(L), np.nanmean(L)
    L = np.where(L >= mean - std, L, np.nan)
    L = np.where(L <= mean + std, L, np.nan)

    B = CIELAB[:, :, 2]
    B = np.where(B != 0, B, np.nan)
    std, mean = np.nanstd(B), np.nanmean(B)
    B = np.where(B >= mean - std, B, np.nan)
    B = np.where(B <= mean + std, B, np.nan)

    # Calculate ITA
    ITA = math.atan2(np.nanmean(L) - 50, np.nanmean(B)) * (180 / np.pi)

    return ITA

def ITA_label(ITA, method):
    """
    Maps an input ITA to a fitzpatrick label given
    a choice method

    Inputs:
        ITA - (float) individual typology angle
        method - (str) 'kinyanjui' or None

    OutputsL
        (int) fitzpatrick type 1-6
    """

    # Use thresholds from kinyanjui et. al.
    if method == 'kinyanjui':
        if ITA > 55:
            return 1
        elif ITA > 41:
            return 2
        elif ITA > 28:
            return 3
        elif ITA > 19:
            return 4
        elif ITA > 10:
            return 5
        elif ITA <= 10:
            return 6
        else:
            return None
    
    # Use empirical thresholds
    else:
        if ITA >= 45:
            return 1
        elif ITA > 28:
            return 2
        elif ITA > 17:
            return 3
        elif ITA > 5:
            return 4
        elif ITA > -20:
            return 5
        elif ITA <= -20:
            return 6
        else:
            return None

class VGG16WithMetadata(nn.Module):
    def __init__(self, vgg_model, num_classes, num_metadata_features=1):
        super(VGG16WithMetadata, self).__init__()
        # Image feature extraction (VGG16 without classifier)
        self.features = vgg_model.features
        # Separate paths for image and metadata
        self.image_classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 256)
        )
        # Metadata processing path
        self.metadata_processor = nn.Sequential(
            nn.Linear(num_metadata_features, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        # Combined classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, metadata):
        # Process image through VGG features
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.image_classifier(x)
        # Process metadata
        metadata = self.metadata_processor(metadata)
        # Combine features
        combined = torch.cat((x, metadata), dim=1)
        # Final classification
        return self.final_classifier(combined)



def load_model(model_path, num_classes):
    """Load the trained VGG16 model"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model architecture
    base_model = models.vgg16()

    model = VGG16WithMetadata(
        vgg_model=base_model,
        num_classes=num_classes,
        num_metadata_features=1  # Fitzpatrick scale
    )
    
    # Load the trained weights
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

def preprocess_image(image_path):
    """Preprocess a single image following the same transforms as training"""
    # Read image
    image = io.imread(image_path)
    
    # Convert grayscale to RGB if needed
    if len(image.shape) < 3:
        image = skimage.color.gray2rgb(image)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension


def predict_image(model, image_tensor, fitz, device):
    """Make prediction for a single image and return human-readable category"""
    categories = ['malignant', 'benign', 'non-neoplastic']
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        metadata_tensor = torch.tensor([[float(fitz)]], dtype=torch.float).to(device)

        outputs = model(image_tensor.float(), metadata_tensor)
        probabilities = torch.exp(outputs)
        
        # Get the highest probability prediction
        top_prob, top_label = torch.max(probabilities, 1)
        
        category = categories[top_label.item()]
        confidence = top_prob.item() * 100
        
        return category, confidence

def main():
    # Configuration
    model_path = '/teamspace/studios/this_studio/pth/model_path_15_high_expert_select.pth'
    num_classes = 3
    
    # Load model
    print("Loading model...")
    model, device = load_model(model_path, num_classes)
    
    # Example usage
    image_path = '/teamspace/studios/this_studio/hash_image/b8bf2ccc8ceeb97d4bc30c16d536834f.jpg'
    print(f"Processing image: {image_path}")
    
    fitz = 2

    # Preprocess and predict
    image_tensor = preprocess_image(image_path)
    category, confidence = predict_image(model, image_tensor, fitz, device)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Category: {category}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()