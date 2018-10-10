import csv as csv
import cv2 as cv2
import numpy as np
import pickle as pkl
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


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
        measurements = pkl.load( open( "pickles/measurements.pkl", "rb" ) ) 
    else:
        lines = [ ]
        images = [ ]
        steers = [ ]
        throts = [ ]

        for path in paths:

            with open( path + 'driving_log.csv' ) as csvfile:

                reader = csv.reader( csvfile )

                for line in reader:

                    files = [ ]
                    steering_angles = [ ]
                    throttle_values = [ ]

                    files.append( path + 'IMG/' + line[0].split( '/' )[-1] )
                    files.append( path + 'IMG/' + line[1].split( '/' )[-1] )
                    files.append( path + 'IMG/' + line[2].split( '/' )[-1] )

                    steering_angles.append( float( line[3] ) )
                    steering_angles.append( steering_angles[0] + correction )
                    steering_angles.append( steering_angles[0] - correction )
                    
                    throttle_values.append( float( line[4] ) )

                    for file, i in zip( files, range( 3 ) ):
                        images.append( preprocess_image( file ) )
                        measurements.append( steering_angles[i] )
                        images.append( cv2.flip( images[-1], 1 ) )
                        measurements.append( steering_angles[i] * -1.0 )
                    
                    lines.append( line )

        images = np.array( images )
        measurements = np.array( measurements )
        
        if save_pickles:
            pkl.dump( lines, open( "pickles/lines.pkl", "wb" ) )
            pkl.dump( images, open( "pickles/images.pkl", "wb" ) )
            pkl.dump( measurements, open( "pickles/measurements.pkl", "wb" ) )
            pkl.dump( correction, open( "pickles/correction.pkl", "wb" ) )

    return lines, images, measurements;


def train_model_0( x, y, epochs=11, save_model=True, model_name='model' ):

    model = Sequential( )

    model.add( Lambda( lambda x: x / 255.0 - 0.5, input_shape=( 66, 200, 3 ) ) )
    
    model.add( Convolution2D( 24, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu" ) )
    model.add( Convolution2D( 36, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu" ) )
    model.add( Convolution2D( 48, 5, 5, subsample=(2, 2), border_mode='valid', activation="relu" ) )
    
    model.add( Convolution2D( 64, 3, 3, border_mode='valid', activation="relu" ) )
    model.add( Convolution2D( 64, 3, 3, border_mode='valid', activation="relu" ) )
    
    #model.add( Convolution2D( 6, 5, 5, activation="relu" ) )
    #model.add( MaxPooling2D( ) )

    #model.add( Convolution2D( 6, 5, 5, activation="relu" ) )
    #model.add( MaxPooling2D( ) )

    model.add( Flatten( ) )
    
    model.add( Dense( 100 ) )
    model.add( Dense( 50 ) )
    model.add( Dense( 10 ) )
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
steering_correction_factor = 0.22

lines, x_train, y_steer, y_throttle = load_image_data( paths, steering_correction_factor, False, False )

print( x_train.shape )
print( y_steer.shape )
print( y_throttle.shape )

train_model_0( x_train, y_steer, 3, True, 'test_steer' )
train_model_0( x_train, y_throttle, 3, True, 'test_throttle' )

#Center Image [0], Left Image [1], Right Image [2], Steering [3], Throttle [4], Brake [5], Speed [6]

# LOG:

# first attempt with Nvidia architecture, using ReLU activation with no dropout or pooling
# car travels completely around the track
# Epoch 3/3
# 44520/44520 [==============================] - 34s - loss: 0.0088 - val_loss: 0.1889

# fixed below, BGR 2 YUV when training, RGB 2 YUV when driving, car stays centered until 
# curve after bridge
# Epoch 3/3
# 44520/44520 [==============================] - 68s - loss: 0.0049 - val_loss: 0.2245

# with preprocessing refactor, there should be no improvement, car fails spectacularly
# car seems to track race lines from outside of the track
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