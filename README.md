Video Finger Lock
Using the number of fingers raised in a video display,
allow users to input a password to unlock a virtual lock.

Sources of inspiration I used for my approach:
- https://gogul09.github.io/software/hand-gesture-recognition-p1
- https://gogul09.github.io/software/hand-gesture-recognition-p2
- https://dev.to/amarlearning/finger-detection-and-tracking-using-opencv-and-python-586m
- https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html (OpenCV's docs in general were often referenced)
- https://www.pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/

Step 1: Domain Engineering
It was a lot of tweaking to get a video feed that could identify finger counts
with even a moderate degree of consistency. I did a couple of things to
ease this process.
1. Do the project in my room, shades closed (though my shades are not black-out
   so there still was some leakage), light from across the room pointed at my hand
2. Take a green-blue t-shirt and put it around a pillow used as the backdrop
3. Prop the computer up on a few textbooks to get a good angle
4. Wear a blue sweatshirt while testing to make sure my hand is more
   distinguishable
The rest of the background tended to include some specks of light that were
picked up as false impressions, but generally were able to be worked around.
My face is also often on the side of the image as well, but because my face
wasn't right near/obscured by my hand this caused no interference.
For my technical environment I used my 2016 MacBook Pro running macOs Mojave.
I did the assignment in Python heavily taking advantage of OpenCV and also using
NumPy a bit and then some standard Python libraries (os, math, etc.).
I didn't actually use a "library" - because I went with the medium of Video
instead of Images I would do an example, attempt to perform the sequence,
capture the output images (both the normal frame captured and the machine-generated
frame used for decision making), then report the results afterwards.
So my "library" was generated through testing. But these are what each test
attempted to cover:
1. (examples/attempt-1560686527.26) Requested sequence: first in center
   then splay in corner. In the context of this project, that corresponds
   to 0 fingers in the Center, then 5 fingers in the Top-Right
2. (examples/attempt-1560688006.59) Requested sequence again.
   (Center, 0), (Top-Right, 5)
3. (examples/attempt-1560688184.06) False negative.
   Should have been (Center, 0), (Top-Right, 5) but found (Center, 0), (Top-Right, 1)
   This can likely be considered a user error due to not having fingers fully
   in the display, but it would be nice if this were adequately handled.
4. (examples/attempt-1560688236.64) False negative.
   Should have been (Center, 0), (Top-Right, 5) but found (Center, 4), (Top-Right, 5)
   This is due to there being many shadows on the fist when the knuckles face
   towards the camera - the way I was able to get the fist to work was through
   turning it sideways.
5. (examples/attempt-1560688310.07) False positive.
   Should have been (Center, 1), (Top-Right, 5) but found (Center, 0), (Top-Right, 5)
   Cases like this (either as false negatives or false positives) happened often
   because the read value often fluctuated up and down raidly. For a production
   system, some sort of smoothing would need to be in place. So, for this
   example, the particular frame happened to capture a 0 (the password's value)
   even though I'm clearly displaying a 1.
6. (examples/attempt-1560688453.51) False positive.
   Should have been (Center, 0), (Top-Right, 4) but found (Center, 0), (Top-Right, 5)
   This happened for the same reason as above - fluctuations in the finger count.
7. (examples/attempt-1560688698.56)
   (Center, 0), (Center, 3)
8. (examples/attempt-1560689100.69)
   (Center, 0), (Center, 3), (Center, 5)
9. (examples/attempt-1560689205.04)
   (Center, 0), (Bottom-Right, 0), (Top-Center, 0)
10. (examples/attempt-1560689610.42)
    (Top-Left, 1), (Top-Center, 2), (Top-Right, 3)
11. (examples/attempt-1560689801.83)
    (Center-Left, 1), (Center, 2), (Center-Right, 3)
12. (examples/attempt-1560689908.17)
    (Top-Left, 0), (Center, 2), (Top-Right, 5), (Center, 3), (Bottom-Center, 0)
These captured images were stored in jpeg format and the output result file
is in txt format. The input format was, of course, the video frames as captured
by my Mac's built-in camera.
I definitely could have done a better job of Domain Engineering my space. My
system would have been much less buggy if I were more thorough about clearing
all semblances of red from my background - which would have lead to more
aggressively tuned thresholds so likely more consistently accurate numbers. I
think lighting could also have been done much better - all I did was point a
light more directly at me from across the room but tweaking that might have
improved things even more.
Using a video feed instead of photos greatly increased my development speed
though because, in a single run of my program, I could test out many differences
and get immediate feedback.

Step 2: Vocabulary
My system follows naturally from our human understanding - it simply counts
the number of extended fingers. I think the system would be nicer if there
was not a movement component, but it was required to fulfill the basic spec
so I included one that breaks space into 9 sections - Top Left, Top Center,
Top Right, Center Left, Center, Center Right, Bottom Left, Bottom Center,
and Bottom Right. So, in total, there are 54 unique combinations (6 finger
positions of 0 through 5 and 9 spatial locations).
I think using number of raised fingers makes sense because many people grow up
learning to count on their fingers so the positions are pretty natural to us.
I even tested some small cultural differences (ex: extending thumb to count "1"
like done in Europe vs extending the pointer finger) and my system held up to
the task, so it might also be effective cross culturally. This also helps avoid
problems like some culture having an obscene gesture be part of the vocabulary
(like a backward peace sign being a way to flip people off in Australia).
Like I said before, the spatial component was a bit more of an afterthought
meant to give me the flexibility to complete the assignment as listed, but I
think some additional tuning would make it nicer. Instead of splitting the space
in front of the computer into thirds it should probably be like Left 2/5ths,
Center 1/5th, Right 2/5ths as well as Top 2/5ths, Center 1/5th, Bottom 2/5ths.
The Central areas tended to be easy to steady oneself in and it would be nice
to not have to exagerate any movement in order to transition to the next location,
so I'll keep this in mind for future projects. On top of that, the environment
I designed for myself (from step 1) was on my bed and the camera was pointed
at a blue-green covered pillow, so the bottom positions often had me ramming
my hand into the pillow which definitely isn't a good user experience. So there's
some work to do on this component.
The images and their labels are mentioned above in my Step 1 description and
can be found in their full form in the examples directory. Each image has
the labels laid on top of it so you can see how the results are generated.
The Machine View is also very useful. It turns the display black and white with
the geospatial lines drawn and the encolsing circle surrounding the hand. This
is how the computer is processing the image, so might be useful to you as a
grader for evaluating my work.
I think my printing text with the labels over the camera feed was an excellent
decision because it gave feedback to users so they can tweak their hands to the
right position. I also think an interesting option (and one that I used
frequently during my testing) is to display the Machine View instead of the
standard camera view. This made fiddling with my hand until it worked much
easier (while also showing just how truly buggy and jumpy my system is).
I guess this is also the right section for talking about how I arrived at my
labels for each image (captured via video). This is the overall approach:
1. Capture an image from the video camera.
2. Parse out all of the red parts of the image over a certain threshold.
3. Identify the largest contour (collection of red) in the image.
4. Draw a bounding rectangle around that contour and note the center coordinates,
   width, and height.
5. Draw a circle with radius based on the smaller of the height or the width.
6. Create a new frame that is just the circle from step 5 and find the bitwise
   difference between it and the original image. Now we have contours that
   should indicate how many fingers are being raised.
7. Do a bit of cleanup (discount the wrist, remove contours that are particularly
   long where the circle happens to cross part of the hand)
8. Compare the x, y coordinates found in the center of the rectangle to
   a coordinate system that cuts the spatial region into approx. thirds
   to determine where the hand is.
9. The user will see the output labels on top of each frame, so they can press
   'f' to capture that frame.
10. After all desired gestures are captured, users can press 'j' to submit
    their frames to be compared against the original password.
Overall I think this approach worked pretty well. A couple things I came across
on the internet or thought of myself but didn't fully get to play with:
- Using a histogram to get hand color to automatically tune to local lighting
  conditions
- Adding a delay between readings so users have time to press 'f' if the
  desired labels have been produced. I definitely ran into this issue a lot
  during my testing mostly because my solution is jumpy and buggy and prone
  to constantly shifting categorization
- Using deep learning to identify finger count. This is an area it would shine:
  just 6 buckets to pick from and plenty of pictures of hands to be possibly
  captured. Then the system would handle shadows and strange orientations much
  better. For example, a closed fist with the fingers pointing to the computer
  (a common way to show a "fist" or "zero") didn't work because of the shadows
  so the side of the fist needed to be shown. A more complex solution might
  have handled this better. Or just better domain engineering maybe?

Step 3: Grammar
My grammar is fairly simple without any reseting and with still leveraging
the keyboard for some input (mainly "capture this image as part of the
sequence" and "test the input sequence against the password"). But I think
that combining keyboard input and video camera input isn't inherently a bad
thing - it is likely that the most ideal system will have many ways to interact
with it that flow together to make the most optimal flow.
I think the successes I have listed in examples/ are representative of ways
real users might use the system. And the failures are explained in Step 1,
so I won't go over that again. In general, the biggest challenge for the system
is that the methods it uses are fairly crude. Because of the millions of causes
of interference for the system (ambient light, floating dust particles, light
coming off the computer, tiny camera malfunctions, etc.) it doesn't hold up
super well to real usage. It would certainly be frustrating for a "real user"
to use. But, it's still really cool that it often has just about the right
number of fingers that it counts - normally rapidly fluctuating between the
correct value and one below or above it. With some more sophisticated underlying
algorithms and data pre-processing, I think this could be a very useful system.
In terms of how I selected my successes and failures:
For my successes, I just thought up different sequences and tried to input them.
Normally it took a few tries to get it right, but over the short time that I
was generating them I think I was getting better. The last, longest one (5
different position/finger count combinations) I actually did on the first try.
For the failures, I took the base example and tweaked it to either give a
false positive or false negative. The false negatives were pretty easy - due to
the system constantly fluctuating between the correct and incorrect reading I
just had to hit 'f' when the incorrect value was being shown. I made sure to use
an empty fist pointing forwards to give a false negative for the "0" because
I had already noticed this is a case handled poorly by my system. For false
positives I used an adjacent value and waited until the system misclassified
to give the false positive response. As foreshadowed in the project spec, making
these errors occur was not particularly challenging.

Step 4: Enhancements
I made the following enhancements to my system:
1. Use a video feed instead of images. All the images you see in examples/
   were captured when I submitted an image to be checked by my system, so it
   was all processed live. I think this enhancement overall made my project
   much easier because I could test on a whole assortment of examples by
   just moving my hand around, so picking thresholds and the like was easier.
2. Expand the vocabulary to include not just "fist" and "splay", but interpret
   these values as numbers (0 and 5 respectively) and include all the other
   numbers as well. Also, instead of just handling "center" and "corner", split
   the spatial "where" component into 9 sections as explained above.
I thought these two extensions brought me to a good sweet spot. I'm really
happy I was able to experiment with video because I think I might want to
design a system that works with live video for my final project and I liked
extending the vocabulary in the way I did because counting by finger is
something that humans do so naturally. And I found the workload to be pretty
reasonable while still juggling my social life and professional commitments
(I'm too old for all-nighters...) so I think it all worked out well!

Thanks for designing an absolutely awesome project. I honestly really enjoyed
this one if only for the excuse of playing with all these fun new tools, so
thank you for assigning it.