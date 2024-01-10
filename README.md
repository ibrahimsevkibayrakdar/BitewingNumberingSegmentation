
  <br>

YOLOv5 🚀 is the world's most loved vision AI, representing <a href="https://ultralytics.com">Ultralytics</a> open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

</div>
<br>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.8.0**](https://www.python.org/) environment, including
[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Test</summary>
Download the <a href="https://drive.usercontent.google.com/download?id=181tCVWdq2MUv35wiDoGmNEp9APeT5-m4&export=download&authuser=0"> bitewingnumbering.pt</a>  should be downloaded from the checkpoint via drive and the model should be thrown into the checkpoints folder, if there is no folder, it should be created.
After determining the test data and the location of the model, the following test steps should be followed.

```python
python segment/predict.py --img 640 --weights path/of/model/bitewing_numbering.pt --source path/of/test/images --hide-conf --agnostic-nms --device 0 --line-thickness 1


```  


