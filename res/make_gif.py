from moviepy.editor import *

clip = ( VideoFileClip( "./videos/video.mp4" ).subclip( 5.5, 15.5 ) )
clip.write_gif( "./project_video_1.gif", fps=10 )

clip = ( VideoFileClip( "./videos/video.mp4" ).subclip( 18.8, 24.9 ) )
clip.write_gif( "./project_video_2.gif", fps=10 )

clip = ( VideoFileClip( "./videos/video.mp4" ).subclip( 35.5, 41.8 ) )
clip.write_gif( "./project_video_3.gif", fps=10 )

clip = ( VideoFileClip( "./videos/video.mp4" ).subclip( 47.7, 73.3 ) )
clip.write_gif( "./project_video_4.gif", fps=10 )

clip = ( VideoFileClip( "./videos/video.mp4" ).subclip( 76.2, 88.8 ) )
clip.write_gif( "./project_video_5.gif", fps=10 )

clip = ( VideoFileClip( "./videos/challenge.mp4" ).subclip( 3.8, 9.1 ) )
clip.write_gif( "./project_video_6.gif", fps=10 )

clip = ( VideoFileClip( "./videos/challenge.mp4" ).subclip( 66.6, 80.0 ) )
clip.write_gif( "./project_video_7.gif", fps=10 )

clip = ( VideoFileClip( "./videos/challenge.mp4" ).subclip( 24.4, 39.9 ) )
clip.write_gif( "./project_video_8.gif", fps=10 )

clip = ( VideoFileClip( "./videos/challenge.mp4" ).subclip( 40, 60 ) )
clip.write_gif( "./project_video_9.gif", fps=10 )


#clip2 = ( VideoFileClip( "./videos/challenge.mp4" ).resize( 0.5 ) )
#clip2.write_gif( "./project_video_2.gif", fps=10 )
