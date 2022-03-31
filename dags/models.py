import torchvision


### Model class
class Model:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    ### Faster R-CNN vanilla model
    def frcnn_vanilla_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            num_classes=self.num_classes
        )
        return model

    ### Faster R-CNN pre-trained model
    def frcnn_pretrained_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = self.num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        )
        return model
