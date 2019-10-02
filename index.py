from train import ImageRegression
from flask import Flask, escape, request
from flask import jsonify

app = Flask(__name__)

# optimizers: SGD or Adam

# multiclass example (default): nn.LogSoftmax + NLLoss + Adam
regressionModel = ImageRegression('/Volumes/media/classification')

# binary classifier example: nn.Sigmoid + BCELoss + Adam

# regression problem example: linear activation function + nn.MSELoss + Adam

@app.route('/')
def doPrediction():
    topk, topclass = classifier.prediction.loadModelAndPredict('/Volumes/media/classification_model_24.pt', '/Volumes/media/classification/test/TOY_WITH_KID/_Volumes_media_classification_toy_with_kid_products3_82_57_1357528_full.jpg')
    return jsonify({ classifier.idx_to_class[topclass.cpu().numpy()[0][i]]: str(topk.cpu().numpy()[0][i]) for i in range(0, len(topclass.cpu().numpy()[0])) })

   
@app.route('/train')
def train():
    regressionModel.train_and_validate()
    
    
if __name__ == '__main__':
    app.run(debug=True)