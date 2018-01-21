# 3D human pose estimation for photos and videos

## Videos
For the video you have the gif version and the mpg version:<br />
-If we apply a continuity constraint with small weight<br />
![Alt text](continue.gif)
<br />
-If we don't apply a continuity constraint with small weight
<br />
![Alt text](no_continue.gif "If we don't apply a continuity constraint with small weight")
<br />
-If we don't change the weights 
<br />
![Alt text](large.gif "Large weight")
<br />
We can see that the most important point is the weights. Then the adequate initialization will give this results and we
don't need temporality constraint on the energy(see http://files.is.tue.mpg.de/black/papers/BogoECCV2016.pdf)

## Code

Here you can find some scripts or notebook that has been modified for the extension to videos. The code are not executable because a lot of things are missing(caffe, smpl library, models...).


The code comes from :
-https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation
-http://smplify.is.tue.mpg.de/

So each modified files have a different licenses and these modified files follow the license of the original file.
These modifications have been done for a class project.

You can also find the original videos at :
-https://www.dropbox.com/s/5608fx0p23jdvjp/manipulation_videos.zip?dl=0
And the used video is barbell_0002.

To have video from frames ffmpeg has been used.
