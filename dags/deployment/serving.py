import os

import cv2
import numpy as np
import torch
import torchvision

import config


### Inference
class Serving:
    def __init__(self, image_data):
        self.image_data = image_data

        self.num_classes = config.NUM_CLASSES

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device: {}'.format(self.device))

    ### Load model
    def load_model(self):
        # load model (Faster RCNN pre-trained model)
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, self.num_classes
            )
        )

        self.model.load_state_dict(
            torch.load(
                os.path.join(config.PATH_MODELS, 'saved_model.pth'), map_location=self.device
            )
        )
        self.model.to(self.device)

    ### Transform
    def transform(self):
        self.image_tensor = torchvision.transforms.ToTensor()(
            self.image_data
        ).unsqueeze_(0)

    ### Inference execution
    def execution(self):
        self.model.eval()
        with torch.no_grad():
            self.image_tensor = self.image_tensor.to(self.device)
            output = self.model(self.image_tensor)

        output_image = self.image_tensor.cpu().numpy()[0].copy()
        output_image = np.transpose(output_image * 255.0, (1, 2, 0))
        output_image = np.ascontiguousarray(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

        # Draw bounding boxes
        for i, box in enumerate(output[0]['boxes']):
            if output[0]['scores'][i].item() > config.SCORE_THRESHOLD:
                cv2.rectangle(
                    output_image,
                    (int(box[0].item()), int(box[1].item())),
                    (int(box[2].item()), int(box[3].item())),
                    (0, 0, 255),
                    1,
                )
                cv2.putText(
                    output_image,
                    '{:.2f}'.format(output[0]['scores'][i].item()),
                    (int(box[0].item()), int(box[3].item())),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 0),
                    1,
                )

        # Record scores
        output_score = {}
        for key, value in output[0].items():
            output_score[key] = value.cpu().tolist()

        return (output_image, output_score)
