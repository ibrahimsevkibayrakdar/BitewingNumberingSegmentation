
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

```python
python segment/predict.py --img 640 --weights Local/Desktop/BitewingNumberingSegmentation/checkpoints/bitewing_numbering.pt --source Local/Desktop/radiography/test/images --hide-conf --agnostic-nms --device 0 -- --line-thickness 1

</details>
<details>
```  
 ### Checkpoint

Model                                                                                          
 [bitewingsegnumbering.pt](https://drive.usercontent.google.com/download?id=181tCVWdq2MUv35wiDoGmNEp9APeT5-m4&export=download&authuser=0)    

