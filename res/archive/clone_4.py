import csv as csv
import cv2 as cv2
import numpy as np
import pickle as pkl
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2


def show_image( image ):
    # Show a specific image with cv2
    plt.imshow( cv2.cvtColor( image, cv2.COLOR_YUV2RGB ) )
    #plt.imshow( image )
    plt.show( 1 )


def preprocess_image( path ):
    image = cv2.imread( path )[60:140,:,:]
    image = cv2.GaussianBlur( image, ( 3, 3 ), 0 )
    image = cv2.resize( image, ( 200, 66 ), interpolation=cv2.INTER_AREA )
    image = cv2.cvtColor( image, cv2.COLOR_BGR2YUV ) 
    return image


def load_image_data( paths, correction=0.2, save_pickles=False, force_reload=False ):
    
    try:
        open( "pickles/correction.pkl" )
        prior_correction = pkl.load( open( "pickles/correction.pkl", "rb" ) )
        print( prior_correction )
    except IOError:
        prior_correction = -1
    
    if( prior_correction == correction and not force_reload ): 
        lines = pkl.load( open( "pickles/lines.pkl", "rb" ) ) 
        images = pkl.load( open( "pickles/images.pkl", "rb" ) ) 
        steers = pkl.load( open( "pickles/steering.pkl", "rb" ) ) 
        throts = pkl.load( open( "pickles/throttle.pkl", "rb" ) ) 
        speeds = pkl.load( open( "pickles/speeds.pkl", "rb" ) ) 

    else:
        lines = [ ]
        images = [ ]
        steers = [ ]
        throts = [ ]
        speeds = [ ]

        for path in paths:

            with open( path + 'driving_log.csv' ) as csvfile:

                reader = csv.reader( csvfile )
                next( reader )
                
                for line in reader:

                    files = [ ]
                    steering_angles = [ ]

                    files.append( path + 'IMG/' + line[0].split( '/' )[-1] )
                    files.append( path + 'IMG/' + line[1].split( '/' )[-1] )
                    files.append( path + 'IMG/' + line[2].split( '/' )[-1] )

                    steering_angles.append( float( line[3] ) )
                    steering_angles.append( steering_angles[0] + correction )
                    steering_angles.append( steering_angles[0] - correction )
                    
                    throttle_value = float( line[4] )
                    speed_value = float( line[6] )

                    for file, i in zip( files, range( 3 ) ):
                        images.append( preprocess_image( file ) )
                        steers.append( steering_angles[i] )
                        throts.append( throttle_value )
                        speeds.append( speed_value )
                        
                        if abs( steering_angles[i] ) > 0.011:

                            images.append( cv2.flip( images[-1], 1 ) )
                            steers.append( steering_angles[i] * -1.0 )
                            throts.append( throttle_value )
                            speeds.append( speed_value )
                    
                    lines.append( line )

        images = np.array( images )
        steers = np.array( steers )
        throts = np.array( throts )
        speeds = np.array( speeds )
        
        if save_pickles:
            pkl.dump( lines, open( "pickles/lines.pkl", "wb" ) )
            pkl.dump( images, open( "pickles/images.pkl", "wb" ) )
            pkl.dump( steers, open( "pickles/steering.pkl", "wb" ) )
            pkl.dump( throts, open( "pickles/throttle.pkl", "wb" ) )
            pkl.dump( speeds, open( "pickles/speeds.pkl", "wb" ) )
            pkl.dump( correction, open( "pickles/correction.pkl", "wb" ) )

    return lines, images, steers, throts, speeds;


def train_model_0( x, y, epochs=11, save_model=True, model_name='model' ):

    model = Sequential( )

    model.add( Lambda( lambda x: x / 255.0 - 0.5, input_shape=( 66, 200, 3 ) ) )
    
    model.add( Convolution2D( 24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.0011) ) )
    model.add( ELU( ) )
    
    model.add( Convolution2D( 36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.0011) ) )
    model.add( ELU( ) )
    
    model.add( Convolution2D( 48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.0011) ) )
    model.add( ELU( ) )
    
    model.add( Convolution2D( 64, 3, 3, border_mode='valid', W_regularizer=l2(0.0011) ) )
    model.add( ELU( ) )
    
    model.add( Convolution2D( 64, 3, 3, border_mode='valid', W_regularizer=l2(0.0011) ) )
    model.add( ELU( ) )

    model.add( Flatten( ) )
    
    model.add( Dense( 100, W_regularizer=l2(0.0011) ) )
    model.add( ELU( ) )
    
    model.add( Dense( 50, W_regularizer=l2(0.0011) ) )
    model.add( ELU( ) )
    
    model.add( Dense( 10, W_regularizer=l2(0.0011) ) )
    model.add( ELU( ) )
    
    model.add( Dense( 1 ) )
    
    model.compile( loss='mse', optimizer='adam' )
    
    model.fit( x, y, validation_split=0.2, shuffle=True, nb_epoch=epochs )

    if not save_model:
        return
    
    model.save( './models/' + model_name + '.h5' )


paths = [ \
         './../sim_data/session1/', \
         './../sim_data/session2/', \
         './../sim_data/session3/'  \
        ]
steering_correction = 0.22

lines, x_train, y_steering, y_throttle, y_speed = load_image_data( paths, steering_correction, False, False )

print( x_train.shape )
print( y_steering.shape )
print( y_throttle.shape )
print( y_speed.shape )

train_model_0( x_train, y_steering, 4, True, 'steering' )
#train_model_0( x_train, y_throttle, 3, True, 'throttle' )
train_model_0( x_train, y_speed, 9, True, 'speed' )

#Center Image [0], Left Image [1], Right Image [2], Steering [3], Throttle [4], Brake [5], Speed [6]

# LOG:
# same but with λ decreased to 0.0011
# 
### Steering
# Epoch 4/4
# 41871/41871 [==============================] - 33s - loss: 0.0311 - val_loss: 0.2317
### Speed
# Epoch 9/9
# 41871/41871 [==============================] - 43s - loss: 5.2606 - val_loss: 107.8474

# back to cutting mirrored steering angles < 0.011, only speed-trained, λ increased to 0.001
# terrible, huge step backwards
### Steering
# Epoch 5/5
# 41871/41871 [==============================] - 33s - loss: 0.0459 - val_loss: 0.2849
### Speed
# Epoch 9/9
# 41871/41871 [==============================] - 43s - loss: 8.4778 - val_loss: 105.2397

# removing all images with steering angles < 0.01, with only speed-trained
# car never stayed straight, turned very smoothly, but eventually did some off-roading
### Steering
# Epoch 4/4
# 12536/12536 [==============================] - 13s - loss: 0.0292 - val_loss: 0.0416
### Speed
# Epoch 8/8
# 12536/12536 [==============================] - 13s - loss: 4.6776 - val_loss: 53.2004

# removing mirrored images with steering angles < 0.01, with only speed-trained
# car drove smoother, late turns though indicative of loss of training data
### Steering
# Epoch 4/4
# 41874/41874 [==============================] - 33s - loss: 0.0951 - val_loss: 0.2556
### Speed
# Epoch 8/8
# 41874/41874 [==============================] - 37s - loss: 5.7194 - val_loss: 112.1338

# for all layers: replaced ReLU with ELU, added L2 regularization (λ = 0.001), collecting throttle, speed
# car completes full laps, at a competitive speed.  speed-trained model performed best.
### Steering
# Epoch 5/5
# 44520/44520 [==============================] - 39s - loss: 0.0294 - val_loss: 0.2049
### Throttle
# Epoch 5/5
# 44520/44520 [==============================] - 39s - loss: 0.2016 - val_loss: 0.1741
### Speed
# Epoch 11/11
# 44520/44520 [==============================] - 39s - loss: 4.3709 - val_loss: 104.4184

# first attempt with Nvidia architecture, using ReLU activation with no dropout or pooling
# car travels completely around the track
# Epoch 3/3
# 44520/44520 [==============================] - 34s - loss: 0.0088 - val_loss: 0.1889

# fixed below, BGR 2 YUV when training, RGB 2 YUV when driving
# car stays centered until curve after bridge
# Epoch 3/3
# 44520/44520 [==============================] - 68s - loss: 0.0049 - val_loss: 0.2245

# with preprocessing refactor, there should be no improvement
# car fails spectacularly, seems to track race lines from outside of the track
# Epoch 3/3
# 44520/44520 [==============================] - 68s - loss: 0.0056 - val_loss: 0.2097

# with cropping, epochs reduced to 3, training time reduced, crashes at new curve past bridge
# Epoch 3/3
# 44520/44520 [==============================] - 73s - loss: 0.0192 - val_loss: 0.2005

# with left, right images, crash at same curve, better centering prior
# Epoch 7/7
# 44520/44520 [==============================] - 130s - loss: 0.0030 - val_loss: 0.2274

# with flipped images and flipped measurements, crash at same curve
# Epoch 7/7
# 14840/14840 [==============================] - 42s - loss: 0.0035 - val_loss: 0.2200

# 2 2D Layers, pre-flipped images, crash after first curve
# Epoch 7/7
# 7420/7420 [==============================] - 15s - loss: 0.0035 - val_loss: 0.2287