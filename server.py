from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, Body, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import json
import io
import torch.nn as nn
import requests
from functools import lru_cache
import os
from waste_guide_data import WASTE_PROCESSING_GUIDE
from trash_bin_location import TRASH_BIN_LOCATIONS
from dotenv import load_dotenv
from openai import OpenAI
import json
import uuid
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union

# Load variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 

client = OpenAI(api_key=OPENAI_API_KEY)

@lru_cache(maxsize=100)
def generate_waste_guide_with_ai(waste_type_vn, waste_type_en):
    """
    Tạo hướng dẫn xử lý rác chi tiết từ dữ liệu cứng nếu có, nếu không thì gọi API OpenAI.
    """
    # Nếu có hướng dẫn sẵn -> trả về luôn
    if waste_type_en in WASTE_PROCESSING_GUIDE:
        return {**WASTE_PROCESSING_GUIDE[waste_type_en], 'generated_by': 'static_data'}

    # Nếu không có thì gọi AI
    try:
        prompt = f"""
        Bạn là chuyên gia về quản lý chất thải và môi trường. Hãy tạo hướng dẫn chi tiết 
        về cách xử lý loại rác: {waste_type_vn} ({waste_type_en}).

        Hãy trả lời theo định dạng JSON với các thông tin sau:
        - category
        - disposal_method
        - preparation
        - recycling_process
        - environmental_impact
        - tips
        - alternatives

        Trả lời bằng tiếng Việt, phù hợp với điều kiện Việt Nam.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia môi trường, trả lời bằng JSON hợp lệ."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        guide_text = response.choices[0].message.content
        if guide_text.startswith('```json'):
            guide_text = guide_text.replace('```json', '').replace('```', '')
        
        guide_data = json.loads(guide_text.strip())
        guide_data['generated_by'] = 'openai'
        return guide_data

    except Exception as e:
        return {
            "category": "Cần phân loại",
            "disposal_method": "Tham khảo hướng dẫn địa phương",
            "preparation": ["Liên hệ cơ quan môi trường để được hướng dẫn"],
            "recycling_process": "Đang cập nhật thông tin",
            "environmental_impact": "Cần xử lý đúng cách để bảo vệ môi trường",
            "tips": ["Luôn phân loại rác đúng cách"],
            "alternatives": ["Tìm hiểu các sản phẩm thay thế thân thiện môi trường"],
            "generated_by": "fallback",
            "error": f"Lỗi AI API: {str(e)}"
        }

# Định nghĩa các model Pydantic
class GuideRequest(BaseModel):
    waste_type_vn: str
    waste_type_en: str

class LocationBase(BaseModel):
    id: str
    name: str
    description: str
    latLng: Dict[str, float]
    category: str

class LocationResponse(LocationBase):
    point: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: str
    parent_class: str
    prediction_vn: str
    class_id: int
    request_id: str

class GuideResponse(BaseModel):
    waste_type_vn: str
    waste_type_en: str
    processing_guide: Dict[str, Any]

app = FastAPI(title="Waste Classification API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 24  # Số lớp của dataset

# Định nghĩa hàm khởi tạo model
def get_model(name, num_classes):
    if name == 'resnet18':
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet50':
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet101':
        from torchvision.models import ResNet101_Weights
        model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'resnet152':
        from torchvision.models import ResNet152_Weights
        model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'densenet121':
        from torchvision.models import DenseNet121_Weights
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == 'densenet169':
        from torchvision.models import DenseNet169_Weights
        model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == 'densenet201':
        from torchvision.models import DenseNet201_Weights
        model = models.densenet201(weights=DenseNet201_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == 'mobilenetv2':
        from torchvision.models import MobileNet_V2_Weights
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'vgg16':
        from torchvision.models import VGG16_BN_Weights
        model = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == 'vgg19':
        from torchvision.models import VGG19_BN_Weights
        model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == 'efficientnet_b0':
        from torchvision.models import EfficientNet_B0_Weights
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'efficientnet_b3':
        from torchvision.models import EfficientNet_B3_Weights
        model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'efficientnet_b7':
        from torchvision.models import EfficientNet_B7_Weights
        model = models.efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Model not supported")
    return model

# Khởi tạo model và mapping
model = None
label_mapping = None

# Vietnamese translation mapping
vietnamese_mapping = {
    "Paper_box": "Hộp giấy",
    "Paper_other": "Giấy khác",
    "Plastic_box": "Hộp nhựa",
    "Plastic_cups": "Cốc nhựa",
    "Metal_package": "Bao bì kim loại",
    "Metal_other": "Kim loại khác",
    "Glass_bottle": "Chai thủy tinh",
    "Glass_other": "Thủy tinh vỡ",
    "Fabric_leather": "Vải và da",
    "Wood_household": "Đồ gỗ gia dụng",
    "Rubber_toy": "Đồ chơi cao su",
    "Rubber_other": "Cao su khác",
    "Electrical_small": "Thiết bị điện nhỏ",
    "Electrical_large": "Thiết bị điện lớn",
    "Food_leftover": "Thức ăn thừa",
    "Food_other": "Thực phẩm khác",
    "Hazardous_other": "Chất độc hại khác",
    "Hazardous_medical": "Chất độc hại y tế",
    "Hazardous_light": "Đèn độc hại",
    "Hazardous_battery": "Pin độc hại",
    "Bulky_wood": "Gỗ cồng kềnh",
    "Other_house_organic": "Chất hữu cơ gia đình khác",
    "Other_household": "Đồ gia dụng khác",
    "Other_plastic": "Nhựa khác"
}

# Định nghĩa transform
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def init_app():
    global model, label_mapping
    model_name = 'densenet201'
    model = get_model(model_name, num_classes)
    # get path model
    model_path = os.path.join(os.path.dirname(__file__), 'models', f'{model_name}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please ensure the model is downloaded and placed in the correct directory.")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    if model_name == 'densenet201':
        with open('label_mapping_densenet.json', 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
    else:
        with open('label_mapping.json', 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)

# Initialize model on startup
@app.on_event("startup")
def startup_event():
    init_app()

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Check if file was uploaded
    if not image:
        raise HTTPException(status_code=400, detail="Không tìm thấy ảnh")
    
    # Read and process the image
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
    
    predicted_class = label_mapping[str(pred)]
    
    # Extract parent class (first number in the label)
    parent_class = predicted_class.split('_')[0]
    
    # Extract name without number prefix
    name_parts = predicted_class.split('_')[1:]
    prediction = "_".join(name_parts)
    
    prediction_vn = vietnamese_mapping.get(prediction, "Chưa có bản dịch")
    
    # Generate AI guide directly in the prediction endpoint
    ai_guide = generate_waste_guide_with_ai(prediction_vn, prediction)
    
    
    # Return immediate prediction response with a unique request ID
    return {
        'prediction': prediction,
        'parent_class': parent_class,
        'prediction_vn': prediction_vn,
        'class_id': pred,
        'processing_guide': ai_guide,
        'generated_by_ai': True,
        'request_id': str(uuid.uuid4())
    }

@app.get("/guide/{waste_type_en}")
async def get_waste_guide(waste_type_en: str, waste_type_vn: str = None):
    if not waste_type_vn:
        waste_type_vn = vietnamese_mapping.get(waste_type_en, waste_type_en)
        
    ai_guide = generate_waste_guide_with_ai(waste_type_vn, waste_type_en)
    
    return {
        'waste_type_en': waste_type_en,
        'waste_type_vn': waste_type_vn,
        'processing_guide': ai_guide
    }

@app.post("/regenerate-guide")
async def regenerate_guide(request: GuideRequest):
    waste_type_vn = request.waste_type_vn
    waste_type_en = request.waste_type_en
    
    if not waste_type_vn or not waste_type_en:
        raise HTTPException(status_code=400, detail="Thiếu thông tin loại rác")
    
    # Clear cache và tạo hướng dẫn mới
    generate_waste_guide_with_ai.cache_clear()
    new_guide = generate_waste_guide_with_ai(waste_type_vn, waste_type_en)
    
    return {
        'waste_type_vn': waste_type_vn,
        'waste_type_en': waste_type_en,
        'guide': new_guide
    }

@app.get("/locations", response_model=List[LocationResponse])
async def get_all_locations():
    """Get all trash bin locations"""
    # For each location, calculate the point from latLng
    locations = []
    for loc in TRASH_BIN_LOCATIONS:
        location = loc.copy()
        location["point"] = {
            "longitude": loc["latLng"]["longitude"],
            "latitude": loc["latLng"]["latitude"]
        }
        locations.append(location)
        
    return locations

@app.get("/locations/category/{category}", response_model=List[LocationResponse])
async def get_locations_by_category(category: str):
    """Get trash bin locations by category"""
    filtered_locations = []
    for loc in TRASH_BIN_LOCATIONS:
        if loc["category"] == category:
            location = loc.copy()
            location["point"] = {
                "longitude": loc["latLng"]["longitude"],
                "latitude": loc["latLng"]["latitude"]
            }
            filtered_locations.append(location)
            
    return filtered_locations

@app.get("/locations/{location_id}", response_model=LocationResponse)
async def get_location_by_id(location_id: str):
    """Get a specific trash bin location by ID"""
    for loc in TRASH_BIN_LOCATIONS:
        if loc["id"] == location_id:
            location = loc.copy()
            location["point"] = {
                "longitude": loc["latLng"]["longitude"],
                "latitude": loc["latLng"]["latitude"]
            }
            return location
            
    raise HTTPException(status_code=404, detail="Location not found")

@app.post("/locations", response_model=LocationResponse, status_code=201)
async def add_location(location: LocationBase):
    """Add a new trash bin location"""
    # Check if ID already exists
    for loc in TRASH_BIN_LOCATIONS:
        if loc["id"] == location.id:
            raise HTTPException(status_code=400, detail="Location ID already exists")
            
    # Add the new location
    location_dict = location.dict()
    TRASH_BIN_LOCATIONS.append(location_dict)
    
    # Return the created location with point calculated
    response = location_dict.copy()
    response["point"] = {
        "longitude": location_dict["latLng"]["longitude"],
        "latitude": location_dict["latLng"]["latitude"]
    }
    
    return response

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <head>
            <title>Phân loại rác thải</title>
        </head>
        <body>
            <h1>Phân loại rác thải</h1>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*">
                <button type="submit">Dự đoán</button>
            </form>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)