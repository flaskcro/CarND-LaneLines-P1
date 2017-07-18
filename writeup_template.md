# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

#### My pipeline consisted of 5 steps. 

- Convert gray scale to image
- Apply Gaussian Blur 
- Apply Canny Edge Detection Algorithm
- Make Interest Region
- Apply hough space 

For the preserve original image, I make a working-image from the original image

```{python}
copy_img = np.copy(image)
```

And I first convert gray scale image from the copy

```{python}
gray_img = grayscale(copy_img)
```

And I apply Gaussian blur filter to the gray scaled image
I choose kernel size 5. that is just got from my experience.
```
blur_img = gaussian_blur(gray_img, 5) 
```

And then I apply Canny algorithm
I choose the parameters following by the Otsu's method

```
def thresholding_otsu(img):
    nbins = 256 # or np.max(img)-np.min(img) for images with non-regular pixel values
    pixel_counts  = Counter(img.ravel())
    counts = np.array([0 for x in range(nbins)])
    for c in sorted(pixel_counts):
        counts[c] = pixel_counts[c]
    p = counts/sum(counts)
    sigma_b = np.zeros((nbins, 1))
    for t in range(nbins):
        q_L = sum(p[:t]) 
        q_H = sum(p[t:]) 
        if q_L ==0 or q_H == 0:
            continue
            
        miu_L = sum(np.dot(p[:t], np.transpose(np.matrix([i for i in range(t)]) )))/q_L
        miu_H = sum(np.dot(p[t:], np.transpose(np.matrix([i for i in range(t, nbins)]))))/q_H
        sigma_b[t] = q_L*q_H*(miu_L-miu_H)**2
        
    return np.argmax(sigma_b)

upper = thresholding_otsu(blur_img)
lower = int(upper/3)
canny_img = canny(blur_img,lower,upper)    
```

I fill the polygon that is my interest region.
I checked the lines and screen and I was adjusting the points values. 
Finally I made my polygon like below

```
ysize = img.shape[0]
xsize = img.shape[1]
width_delta = int(xsize/20)

vertices = np.array([[(150, ysize), (xsize - 50, ysize), (xsize/2 + width_delta, ysize/2 + 50), 
                    (xsize/2 - width_delta, ysize/2 + 50)]], np.int32)

region = region_of_interest(canny_img, vertices)
```

Last step. I apply hough space.
I already got parametes in the last lesson. 

```
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi*5/180 # angular resolution in radians of the Hough grid
threshold = 20     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 15 #minimum number of pixels making up a line
max_line_gap = 5    # maximum gap in pixels between connectable line segments

hough_image = hough_lines(region, rho, theta, threshold, m
```

And I mixed original image and converted image then I save my image to the output directory.

```
result = weighted_img(hough_image, img, α=0.8, β=1., λ=0.)
cv2.imwrite('test_images_output/'+ image, result)
```


![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be that is made in the very stricted condition.
Our road is quite straight, if there is too curved road, my algorithme won't be done well.

And some roads is painted very old and they are too little painted, 
my algoritme will not be done well such kind of roads.

And we adjust interest region by predefined value. If camera locaton and direction are changed, our model is useless.


### 3. Suggest possible improvements to your pipeline

I want to improve my model like below.

 - I define the road line with average from the point, and I got a sloe and intercept the line. I want to improve my model using linear regression, I get all points and I want to slope and intercept from the linear regression algorithm.
 - When the car is on no paitned or less painted road, we should get the lane from the width of road, so I want to mesure the width of road from the pixel. and I want to calculate the each lane of the road.
  