# TheRootOfComputerVision


## Project Overview
The goal of this project was to develop a machine vision program that can keep track of the state of a player's game (fig. 1) board in the game Root. Root is a very complex game with asymmetrical architecture (meaning each player has a unique set of faction specific rules). As such, it is easy to make mistakes especially for newer players. On a high level, the program locates different objects within a player board including various cards and tokens, and stores their location and type. This information can then be applied to check the legality of moves, live during a game. Our MVP is not a fully function checker but provides a jumping off point for future exploration. We decided to focus on Eyrie Dynasties being one of the most dynamic player boards. Our MVP identifies Eyrie Dynasties faction roost tokens and cards distinguished by suit as shown in Figure 1 below. 


<p align="center"><img src="Images/readme_board_image.png"/></p>
<p align="center"><i>Figure 1: Eyrie game board</i></p>


The Eyrie Dynasties decree consists of four possible card locations (in files) with four possible card types. Cards are first identified by suit, then sorted into lists based on their file, Recruit, Move, Battle, and Build (as shown on the player board). As for the roosts, the Eryie Dynasties faction only cares about the quantity, so the program will output an integer corresponding to the number of these tokens detected on the player board. For our MVP, we will assume that the Eyrie Dynasties player board will dominate the photo. 

In addition to the decree and roost detection, we considered a number of stretch goals ranging from detecting item tokens (e.g. the boot in the top left corner), to detecting factions beyond just Eyrie Dynasties and their respective tokens.
- Identify other faction boards
- Identify Eyrie Dynasty leader card
- Identify items

### Learning Goals
- We aim to walk away from this project with experience using specific CV algorithms/techniques, and a strong understanding of how to approach a machine vision problem. 

- We also hope to get a view behind the curtain regarding machine vision algorithms, whether that means writing our own, or exploring how open source algorithms work.

- On a less technical level, we hope to explicitly devote time to–as a team–exploring machine vision project examples and open source software available to us to get a more complete understanding of the resources at our disposal.


## Process Overview
After a little experimentation, we chose to rely entirely on deterministic image manipulation algorithms using openCV. This is possible because of the controlled nature of the player boards and their components. The boards will always be the same size with the same color scheme and organization, a feature that we use to our advantage extensively in this project. Thus, we arrive at one of our first major MVP assumptions: the image perfectly isolates the board from a top down view. We were able to largely diminish the limitations of this assumption as described below.  

### Board Orientation
The initial problem we hoped to tackle was being able to take a photo of the player board from any angle and transform it to a clear, top down view. As the rest of mvp goals revolved around searching different portions of the player board, having a standardized view of that board is integral. This problem ended up taking the majority of our time and was split into two parts. 

- Finding known key points in the raw image to define the current orientation of the board via a three dimensional matrix.
- Applying the matrix transformation to the entire image of the board to construct a top down view.

#### Key Point Detection
For key point detection, the goal is to find a either points of a region that is distinct enough from the rest of the board that it can be recognized, no matter the orientation within the image. For the Eyrie player board, after initial testing and deliberation, we decided on the birdsong section of the board, as seen in figure 2, as our standout region. Our goal ws to use the four corners as the key points for calculating the orientation of the board for the later translation.


<p align="center"><img src="Images/birdsong_cropped.png"/></p>
<p align="center"><i>Figure 2: Birdsong region of Eyrie game board</i></p>


In order to find the four corners of the birdsong region, we relied on a combination of opencv's `findContours` functions and `inRange` function. Starting from the initial image, by providing a lower and upper bound for the `rgb` values, `inRange` returns a binary image with white, or `255`, at regions within that color range and black, or `1`, at those outside the range. Then, with this binary image, `findContours` locates the contours, or regions of color, within the image. The hope is that from this process the biggest contour is the one surrounding the birdsong region. This, however, didn't always end up being the case due to noise in the background from objects such as the birds within the top banner art, as shown in figure 3 below, as well as segmentation of the birdsong region, resulting from the branch of the right side of the box, as seen in figure 2.


<p align="center"><img src="Images/banner.png"/></p>
<p align="center"><i>Figure 3: Eyrie game board banner</i></p>


First, in order to remove the segmentation of the birdsong region, a general blur was preformed on the whole image, using the `resize` function. The effect of blur on the contours can be seen in figures 4 & 5 below.


<p align="center"><img src="Images/bad_birdsong_contour.png"/></p>
<p align="center"><i>Figure 4: Birdsong contour without blur</i></p>

<p align="center"><img src="Images/good_birdsong_contour.png"/></p>
<p align="center"><i>Figure 5: Birdsong contour with blur</i></p>


Alongside this, in order to remove the noise, the process of getting the birdsong contour was split into two steps.

First, we found an even more distinguishable portion of birdsong using a smaller color range. This portion was always able to be identified when relatively consistent lighting was provided. This region can be seen in figure 6 below.


<p align="center"><img src="Images/initial_birdsong_region.png"/></p>
<p align="center"><i>Figure 6: Initial birdsong region</i></p>

Second, We found the full birdsong contour with the original color range, but only in the region around the original distinguished portion. This is performed by creating a rectangle around the region and using opencv's `bitswise_and` function to get only the region of the original image within that rectangle. This search region can be seen in figure 7 below.


<p align="center"><img src="Images/new_search_region.png"/></p>
<p align="center"><i>Figure 7: Birdsong search region</i></p>

By first determining a smaller region in which to search, the process became much more accurate and the final contour is achieved. For the board transformation, however, we specifically need the four corners of this new contour. For this, we utilize the `appx_best_fit_ngon` function within our helper functions file to find the quadrilateral with the smallest area the contains every point within the contour. We found this function through stack overflow and due to the time constraints never dived deep into it's inner workings. This provided us with four points incredibly close to four corners of the contour. Due to the incredibly sensitive nature of the board transformation process, however, we the finally looped the `move_closest_point_toward` function on each of the four points. What this did was shift them closer to the closest point within the actual contour. By performing this final translation, we finally arrived at the four points used for the board transformation. These four points and the box the enclose can be seen in figure 8 below.


<p align="center"><img src="Images/final_box.png"/></p>
<p align="center"><i>Figure 8: Birdsong region bounding box</i></p>

#### Board Transformation
The actual transformation is implemented in the transform_board() function in transform_eyrie_board.py and is largely enabled by an openCV function getPerspectiveTransform(). getPerspectiveTransform() does all the heavy lifting, taking in two sets of four (x, y) points and computing the transformation that moves the first set of points to the second set of points.getPerspectiveTransform() can then be used in conjunction with another function, warpPerspective(), which applies the transformation to an entire image. transform_board() takes in the corners of birdsong as the first four points and the aforementioned openCV functions according to a given set of transformed points, both provided as an argument. Notably, we specifically chose to leave the second set of points as an argument so the same function could be used for other faction boards (in addition to Eyrie Dynasties) which all have the same birdsong rectangle but in a different size. 
It is important to note one particular assumption that transform_board() makes to function correctly. The function cannot assume that the inputted first set of points are in the same order as the second set because the point detection algorithm does not detect where the points are in reference to the rest of the board. Thus, it must reorder the first set of points to match the second set. Essentially, transform_board() begins with the assumption that the farthest left point must be one of the two leftmost corners of the birdsong rectangle, and then, through a series of distance comparisons and height comparisons, reorders the list to be top-left, bottom-left, top-right, bottom-right. The reordering, however, only works if the board is not upside down, otherwise that initial assumption is wrong. Hence, for the transform_board() function to work for any orientation, some initial processing must be done first. 

### Roost Detection
Here, we take advantage of the board transformation to find the roosts based on their size in the transformed image. The function detect_roosts() in roost_detection.py first converts the transformed image into a binary image, highlighting only elements that match the blue color scheme of the roost tokens. It then implements the openCV contour search algorithm which isolates each roost as a unique contour among many many others. Finally, it filters all of the detected contours by area to yield only the roosts and outputs the remaining number of contours. Notably, the roost detection only works if the roosts are mostly in their proper locations so they are sufficiently separated from each other and surrounding blue features.

<p align="center"><img src="Images/roosts.png"/></p>
<p align="center"><i>Figure 10: Roosts color range binary image</i></p>

### Decree Detection
The decree detection uses a similar approach to the roost detection with a little added complexity due to the much larger variety of possible states. The function detect_decree() in decree_detection.py generates four different binary images, highlighting the respective colors of the banners of the four card types: mouse (orange), bunny (yellow), fox (red), and bird (blue). Luckily, the banners are quite pure in color, allowing these binary images very effectively filter out the unrelated features of the board. The effectiveness of this filter is key for the next step. To account for the inconsistent potential positions of each card in the decree (i.e. how close they are together and how aligned they are with each column), we decided that we can't rely on a contour search like we did for the roosts. Instead, we approximate the number of pixels in the card banner for each type and determine the number of cards of that type based on the total number of pixels. 
Column filtering once again relies on the board transformation. Because the boards are a set width, we assume that the transformation can effectively crop the image to align with the edges of the board. This cropping allows the program to filter detected cards by segmenting the image into four parts. It first approximates the center of the board using the inside edge of birdsong, and then approximates the subdivisions measuring from the center. Notably, this would not work if we didn't know that the actual width of each subdivision is approximately constant (due to the designations on the board).

<p align="center"><img src="Images/card_binaries.png"/></p>
<p align="center"><i>Figure 11: Card type color range binary images</i></p>

## Reflection

### Challenges
We encountered a lot of challenged throughout the whole process. The majority of those problems, however, were concentrated around the birdsong region corner detection and transformation of the board, as talked about within those portions of the process review. Within the birdsong region corner detection, we initially ran into trouble in the form of a lot of noise. This was the first roadblock we encountered but we were able to circumvent this through the secondary regional search. Following this we also had issues simply finding the full contour due to the branch within the artwork. This, thankfully, was resolved by blurring the image. Finally we had issues with the translation itself being incredibly off. We eventually found that this was simply due to the sensitivity of the process which led us to shifting four detected corners of the bounding box to closest points of the contour. 

(Insert other challenges from other portions)

Aside from the more specific issues, there are still other more general challenges that are still a challenge. The first and biggest one is lighting. Not having consistent lighting can completely ruin the translation, color detection, and everything else. Alongside that, our model currently can only translate images that are within a certain degree offset range. Alongside that, our model assumes they are always upright so to much variation in player board orientation still ruins the end result.
Most of our major challenges over the course of this project had to do with image inconsistencies. There are a lot of factors we can't control about the provided image, including but not limited to, scale, perspective, resolution, lighting, and exposure. We spent the majority of our time trying to tackle some of these with the board transformation, however, we were not able to tackle the lighting-related issues. Additionally, we spent a lot time wrestling with image resolution. Small details picked up in higher resolution images turned out to be quite problematic for our contour detection, however, low resolution images sacrificed accuracy. Specifically for the transformation, a high level of accuracy was required as a result of the massive scale difference between our reference points (the birdsong corners) and the entire board. We found that small discrepancies in the birdsong corners would have enormous impacts on the full image transformation. 

### Improvements


### Takeaways
