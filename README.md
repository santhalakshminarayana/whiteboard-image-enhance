# Whiteboard image color enhancement in Python
Enhance whiteboard images by applying image-processing techniques. Whiteboard image color enhancement is based on this [ImageMagick command line gist](https://gist.github.com/lelandbatey/8677901)

Converted following ImageMagick commands to **Python** and **OpenCV** by applying enhancement functions like
- Difference of Gaussian (DoG)
- Contrast Stretching
- Gamma Correction

```bash
-morphology Convolve DoG:15,100,0 -negate -normalize -blur 0x1 -channel RBG -level 60%,91%,0.1
```
Run **whiteboard_image_enhance.py** by passing _input_ and _output image path_

```shell
$ python whiteboard_image_enhance.py -i input.jpg -o output.jpg
```

### Results

<table border='0'>
  <tr>
    <th>Original</th>
    <th>Enhanced</th>
  </tr>
  <tr>
    <td><img src='https://github.com/santhalakshminarayana/white-board-enhance/blob/main/images/11.jpg' 
        width='500px' height='300px' /></td>
    <td><img src='https://github.com/santhalakshminarayana/white-board-enhance/blob/main/result/11_o.jpg' 
         width='500px' height='300px' /></td>
  </tr>
  <tr>
    <td><img src='https://github.com/santhalakshminarayana/white-board-enhance/blob/main/images/22.jpg'
             width='500px' height='300px' /></td>
    <td><img src='https://github.com/santhalakshminarayana/white-board-enhance/blob/main/result/22_o.jpg' 
         width='500px' height='300px' /></td>
  </tr>
  <tr>
    <td><img src='https://github.com/santhalakshminarayana/white-board-enhance/blob/main/images/1.jpeg' 
             width='500px' height='300px'/></td>
    <td><img src='https://github.com/santhalakshminarayana/white-board-enhance/blob/main/result/1_o.jpg'
             width='500px' height='300px'/></td>
  </tr>
  <tr>
    <td><img src='https://github.com/santhalakshminarayana/white-board-enhance/blob/main/images/2.jpeg'
             width='500px' height='300px' /></td>
    <td><img src='https://github.com/santhalakshminarayana/white-board-enhance/blob/main/result/2_o.jpg'
             width='500px' height='300px' /></td>
  </tr>
</table>
