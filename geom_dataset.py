import numpy

def create_geom_dataset(size=28):
    data = []
    
    shapes = [numpy.array([[1,1,1,1],
                           [1,1,1,1],
                           [1,1,1,1],
                           [1,1,1,1]]),
              numpy.array([[1,0,0,0],
                           [1,1,0,0],
                           [1,1,1,0],
                           [1,1,1,1]]),
              numpy.array([[0,0,0,1,0,0,0],
                           [0,0,1,1,1,0,0],
                           [0,1,1,1,1,1,0],
                           [1,1,1,1,1,1,1]]),
              numpy.array([[1,0,0,0],
                           [1,1,0,0],
                           [1,1,1,0],
                           [1,1,0,0],
                           [1,0,0,0]]),
              numpy.array([[1,0,0,0,1],
                           [0,1,0,1,0],
                           [0,0,1,0,0],
                           [0,1,0,1,0],
                           [1,0,0,0,1]]),
              numpy.array([[0,0,1,0,0],
                           [0,1,1,1,0],
                           [1,1,1,1,1],
                           [0,1,1,1,0],
                           [0,0,1,0,0]])]
                          

    for s in shapes:
        for x in range(size-s.shape[0]):
            for y in range(size-s.shape[1]):
                z = numpy.zeros((size,size),dtype='float32')
                z[x:x+s.shape[0],y:y+s.shape[1]] = s
                data.append(z.flatten())
    
    return numpy.array(data)
