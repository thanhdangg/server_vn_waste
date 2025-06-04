from flask import Flask, request, jsonify, current_app
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import json
import io
import torch.nn as nn

app = Flask(__name__)

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

def init_app():
    global model, label_mapping
    model = get_model('densenet169', num_classes)
    model.load_state_dict(torch.load("models/densenet169.pth", map_location=device))
    model.to(device)
    model.eval()
    
    with open('label_mapping.json', 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)

# Đảm bảo model được load trước khi xử lý request
@app.before_request
def load_model_if_needed():
    global model
    if model is None:
        init_app()

# Định nghĩa transform
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Không tìm thấy ảnh'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
    
    predicted_class = label_mapping[str(pred)]
    
    return jsonify({
        'prediction': predicted_class,
        'class_id': pred
    })

@app.route('/', methods=['GET'])
def index():
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

if __name__ == '__main__':
    # Khởi tạo model và mapping khi khởi động ứng dụng
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5002)