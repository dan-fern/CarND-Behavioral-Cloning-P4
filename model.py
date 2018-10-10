import csv as csv
import cv2 as cv2
import numpy as np
import pickle as pkl
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2


def show_image( image ):
    plt.imshow( cv2.cvtColor( image, cv2.COLOR_YUV2RGB ) )
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

                        #if abs( steering_angles[i] ) > 0.01:

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

    model.add( Convolution2D( 24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.0005) ) )
    model.add( ELU( ) )

    model.add( Convolution2D( 36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.0005) ) )
    model.add( ELU( ) )

    model.add( Convolution2D( 48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.0005) ) )
    model.add( ELU( ) )

    model.add( Convolution2D( 64, 3, 3, border_mode='valid', W_regularizer=l2(0.0005) ) )
    model.add( ELU( ) )

    model.add( Convolution2D( 64, 3, 3, border_mode='valid', W_regularizer=l2(0.0005) ) )
    model.add( ELU( ) )

    model.add( Flatten( ) )

    model.add( Dense( 100, W_regularizer=l2(0.0005) ) )
    model.add( ELU( ) )

    model.add( Dense( 50, W_regularizer=l2(0.0005) ) )
    model.add( ELU( ) )

    model.add( Dense( 10, W_regularizer=l2(0.0005) ) )
    model.add( ELU( ) )

    model.add( Dense( 1 ) )

    model.compile( loss='mse', optimizer='adam' )

    model.fit( x, y, validation_split=0.2, shuffle=True, nb_epoch=epochs )

    if not save_model:
        return

    model.save( './models/' + model_name + '.h5' )

    return

paths = [ \
#           './../sim_data/udacity/'    \
         ,  './../sim_data/session1/'   \
         ,  './../sim_data/session2/'   \
         ,  './../sim_data/session3/'   \
        ]
steering_correction = 0.22

lines, x_train, y_steering, y_throttle, y_speed = load_image_data( paths, steering_correction, False, False )

print( x_train.shape )
print( y_steering.shape )
print( y_throttle.shape )
print( y_speed.shape )

train_model_0( x_train, y_steering, 4, True, 'model' )

train_model_0( x_train, y_throttle, 3, True, 'throttle' )

train_model_0( x_train, y_speed, 7, True, 'speed' )
