from kserve.model import Model

from ultralytics import YOLO

class Yolov8sModel(Model):
    def __init__(self, name):
        super().__init__(name)
    
    def load(self):
        self.model = YOLO('yolov8s.pt')
        try:
            self.model('https://ultralytics.com/images/bus.jpg')
            self.ready = True
        except:
            print("Model load failed")
            self.ready = False
        return self.ready
    
    def preprocess(self, request, headers = None):
        return request

    def predict(self, request, headers = None):
        # Process request
        instances = request.get('instances')
        for instance in instances:
            try:
                img = instance.get("image")
                if img is None:
                    img = instance.get("url")
                
                # Run inference
                predict = self.model(img)
                
                instance["result"] = []
                
                if predict is None or len(predict) == 0:
                    continue
                
                # Get predictions
                instance['result'] = []
                for result in predict[0].boxes:
                    instance['result'].append({
                        "name": predict[0].names[result.cls.item()],
                        "confidence": result.conf.item(),
                        "xmin": result.xyxy[0][0].item(),
                        "ymin": result.xyxy[0][1].item(),
                        "xmax": result.xyxy[0][2].item(),
                        "ymax": result.xyxy[0][3].item(),
                    })
            except BaseException as e:
                print("infer error")
       
        return request

    
    def postprocess(self, request, headers = None):
        return request


if __name__ == "__main__":
    model = Yolov8sModel("yolov8s")
    model.load()
    

    assert model.ready == True, f"{model.name} is not ready"
    
    # Test inference
    instances = [
        {
            # "image": './bus.jpg',
            "url": "https://ultralytics.com/images/bus.jpg"
        }
    ]
    request = {"instances": instances}
    
    ret = model.preprocess(request)
    ret = model.predict(ret)
    ret = model.postprocess(ret)
    
    print(ret)
    
    # start model server
    # import json
    # from kserve.model_server import ModelServer
    # _server = ModelServer(http_port=5000)
    # model_list = [model]
    # _server.start(model_list)
