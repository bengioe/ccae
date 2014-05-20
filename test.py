
import theano
import theano.tensor as T

def f(a,b):
    print a,b
    return T.dot(a,b)

a = theano.tensor.matrix()
b = theano.tensor.matrix()

c, updates = theano.map(f, [a,b])


f = theano.function([a,b],c)
print f([[1,2,3],[4,5,6],[7,8,9]],
        [[1,2,3],[4,5,6],[7,8,9]])
