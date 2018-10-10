import csv as csv
import cv2 as cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


def show_image( img ):
    # Show a specific image with cv2
    cv2.imshow( 'image', img )
    cv2.waitKey( 0 )
    cv2.destroyAllWindows( )


def train_model_0( x, y, epochs=7, save_model=True ):
    # Epoch 7/7
    # 7420/7420 [==============================] - 15s - loss: 0.0035 - val_loss: 0.2287
    model = Sequential( )
    model.add( Lambda( lambda x: x / 255.0 - 0.5, input_shape=( 160, 320, 3 ) ) )

    model.add( Cropping2D( cropping=( ( 70, 25 ), ( 0, 0 ) ) ) )
    
    model.add( Convolution2D( 6, 5, 5, activation="relu" ) )
    model.add( MaxPooling2D( ) )

    model.add( Convolution2D( 6, 5, 5, activation="relu" ) )
    model.add( MaxPooling2D( ) )

    model.add( Flatten( ) )
    model.add( Dense( 120 ) )
    model.add( Dense( 84 ) )
    model.add( Dense( 1 ) )

    model.compile( loss='mse', optimizer='adam' )
    model.fit( x, y, validation_split=0.2, shuffle=True, nb_epoch=epochs )

    if( save_model == False ):
        return
    
    model.save( 'model.h5' )


lines = [ ]
paths = [ './../sim_data/session1/', \
          './../sim_data/session2/', \
          './../sim_data/session3/'  \
        ]

images = [ ]
measurements = [ ]
steering_correction_factor = 0.2

for path in paths:

    with open( path + 'driving_log.csv' ) as csvfile:
        
        reader = csv.reader( csvfile )
        
        for line in reader:
            
            path_c = path + 'IMG/' + line[0].split( '/' )[-1]
            path_l = path + 'IMG/' + line[1].split( '/' )[-1]
            path_r = path + 'IMG/' + line[2].split( '/' )[-1]
            
            steering_angle_c = float( line[3] )
            steering_angle_l = steering_angle_c + steering_correction_factor
            steering_angle_r = steering_angle_c - steering_correction_factor

            images.append( cv2.imread( path_c ) )
            measurements.append( steering_angle_c )
            images.append( cv2.flip( cv2.imread( path_c ), 1 ) )
            measurements.append( steering_angle_c * -1.0 )
            
            images.append( cv2.imread( path_l ) )
            measurements.append( steering_angle_l )
            images.append( cv2.flip( cv2.imread( path_l ), 1 ) )
            measurements.append( steering_angle_l * -1.0 )
            
            images.append( cv2.imread( path_r ) )
            measurements.append( steering_angle_r )
            images.append( cv2.flip( cv2.imread( path_r ), 1 ) )
            measurements.append( steering_angle_r * -1.0 )
            
            lines.append( line )

x_train = np.array( images )
y_train = np.array( measurements )

print( x_train.shape )
print( y_train.shape )

train_model_0( x_train, y_train, 3, True )

#Center Image [0], Left Image [1], Right Image [2], Steering [3], Throttle [4], Brake [5], Speed [6]

# LOG:

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