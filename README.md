# Zhang's camera calibration

Implementation of Camera calibration method as presented by Zhengyou Zhang from Microsoft in his paper,
[A flexible new technique for camera calibration](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)

### Results

#### Mean reprojection error(pixels) for calibration images before & after optimization

Image | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11 | #12 | #13 |
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---|--- |--- |
Before | 1.36 | 0.84 | 1.38 | 18.22 | 1.09 | 4.55 | 2.90 | 4.19 | 5.30 | 4.86 | 22.52 | 12.02 | 41.11 |
After | 0.35 | 0.49 | 0.84 | 1.14 | 0.18 | 0.44 | 0.08 | 0.21 | 0.34 | 0.29 | 0.62 | 0.86 | 0.88 |


#### Intrinsic matrix and distortion coefficients' estimates:

1. OpenCV reference:
   ```
   camera matrix:
      [[2.042729e+03 0.000000e+00 7.643600e+02]
      [0.000000e+00 2.035016e+03 1.359026e+03]
      [0.000000e+00 0.000000e+00 1.000000e+00]]

   distortion coefficients(k1 and k2):
      [0.290493410 -2.42737867]
   ```

2. Initial estimates:
   ```
   camera matrix:
      [[2.061892e+03 −2.850592e + 00 7.760024e + 02]
      [0.000000e + 00 2.047799e + 03 1.363240e + 03]
      [0.000000e + 00 0.000000e + 00 1.000000e + 00]]

   distortion coefficients:
      [0.0, 0.0]
   ```

3. Optimization without considering distortion coefficients:
   ```
   camera matrix:
      [[2.058478e+03 −1.454986e + 00 7.540409e + 02]
      0.000000e + 00 2.049441e + 03 1.354108e + 03]
      0.000000e + 00 0.000000e + 00 1.000000e + 00]]

   distortion coefficients:
      []
   ```

4. Optimization with distortion coefficients:
   ```
   camera matrix:
      [[2.048509e+03 −1.830419e + 00 7.587442e + 02
      0.000000e + 00 2.040725e + 03 1.345145e + 03
      0.000000e + 00 0.000000e + 00 1.000000e + 00]]

   distortion coefficients:
      [0.17310107 -0.75331797]
   ```


### Steps to run the code

1. Install dependencies
    ```
    python -m venv venv
    source venv/bin/activate  # ./venv/Scripts/activate for windows
    pip install -r requirements.txt
    ```

2. Run script
    ```
    python Wrapper.py
    ```
