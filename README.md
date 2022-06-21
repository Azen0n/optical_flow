Implementation source: https://github.com/scivision/Optical-Flow-LucasKanade-HornSchunck <br>
Video source: https://www.youtube.com/watch?v=wqctLW0Hb_0
## Dependencies
```
pip install -r requirements.txt
```
## Usage
### Lucas–Kanade method
```
lucas_kanade.py [-h] [-r RADIUS] [-v VIDEO]

options:
  -h, --help            show this help message and exit
  -r RADIUS, --radius RADIUS
                        set kernel radius
  -v VIDEO, --video VIDEO
                        set video path
```
Press ```Q``` to close video.
#### Example
```
python lucas_kanade.py -r 2 -v video.mp4
```

### Horn–Schunck method
```
horn_schunck.py [-h] [-a ALPHA] [-i ITERATIONS] [-v VIDEO]

options:
  -h, --help            show this help message and exit
  -a ALPHA, --alpha ALPHA
                        set regularization constant. Larger values lead to a smoother flow
  -i ITERATIONS, --iterations ITERATIONS
                        set number of iterations for laplacian approximation
  -v VIDEO, --video VIDEO
                        set video path
```
Press ```Q``` to close video.
#### Example
```
python horn_schunck.py -a 20 -i 2 -v video.mp4
```
![Alt Text](https://i.imgur.com/NpJ2J2q.gif)