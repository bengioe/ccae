import numpy
import theano
import theano.tensor as T
from math import*

rng = numpy.random.RandomState(42)

#theano.config.compute_test_value = 'warn'

class HiddenLayer:
    def __init__(self,x,n_in,n_out,activation = T.tanh):
        self.x = x
        self.n_in = n_in
        self.n_out = n_out
        k = sqrt(6.0/(n_in+n_out))
        W_vals = numpy.asarray(rng.uniform(-k,k,size=(n_in,n_out)))
        print (n_in,n_out)
        self.W = theano.shared(value=W_vals,name='W')
        self.b = theano.shared(0.25*numpy.ones((n_out,)),name='b')
        self.output = activation(T.dot(self.x,self.W)+self.b)
        self.params = [self.W,self.b]
        
    def recon_from(self, xp, activation=T.nnet.sigmoid, shared=True):
        self.xp = xp
        
        if shared:
            self.Wp = self.W.T
        else:
            W_vals = numpy.asarray(rng.uniform(-k,k,size=(self.n_out,self.n_in)))
            self.Wp = theano.shared(value=W_vals, name="W'")
            self.params.append(self.Wp)
        self.bp = theano.shared(numpy.zeros((self.n_in,)))
        self.recon = activation(T.dot(xp,self.Wp) + self.bp)
        self.params.append(self.bp)

class CCLayer:
    def __init__(self, x, n_in, n_out, version=1):
        self.n_in = n_in
        self.n_out = n_out
        if version == 1:
            self.x = x
            self.n_in = n_in
            self.n_out = n_out
            k = sqrt(6.0/(n_in+n_out))
            W_vals = numpy.asarray(rng.uniform(-k,k,size=(n_in,n_out)))
            print (n_in,n_out)
            self.W = theano.shared(value=W_vals,name='W')
            self.b = theano.shared(0.25*numpy.ones((n_out,)),name='b')
            self.output = T.tanh(T.dot(self.x,self.W)+self.b)
            self.params = [self.W,self.b]

    def recon_from(self, s):
        # s = (bs, n_out)
        self.mu = theano.shared(
            rng.uniform(0,1,size=(self.n_in,self.n_out)),
            name = 'mu')
        D = numpy.zeros((self.n_in,self.n_out,self.n_out))
        for i in range(self.n_in):
            numpy.fill_diagonal(D[i], 1)
        self.D = theano.shared(D, name='D')
        self.params += [self.mu, self.D]
        """
        r = []
        for i in range(self.n_in):
            k = (s-self.mu[i].reshape((1,self.n_out)))
            l = T.dot(k, self.D[i])
            # v = T.exp(-T.dot(l, k.T)).diagonal()
            # but the dot is expensive for nothing since we're only taking the diagonal
            v = T.exp(-T.mul(l, k).sum(axis=1))
            r.append(v)      

        recon = T.as_tensor_variable(r).T"""

        K = s.dimshuffle('x',0,1) - self.mu.dimshuffle(0,'x',1)
        #numpy.sum(a[:,:,:,numpy.newaxis]*b[:,numpy.newaxis,:,:],axis=-2)
        L = T.sum(K.dimshuffle(0,1,2,'x')*self.D.dimshuffle(0,'x',1,2),axis=-2)
        V = T.exp(-T.mul(L, K).sum(axis=2))
        R = V.T
        self.recon = R

class Model:
    def __init__(self, x, n_in, n_hidden, n_cch):
        self.layerH = HiddenLayer(x, n_in, n_hidden,T.nnet.sigmoid)
        self.layerS = CCLayer(self.layerH.output, n_hidden, n_cch)
        self.layerS.recon_from(self.layerS.output)
        self.layerH.recon_from(self.layerS.recon)
        self.recon = self.layerH.recon
        self.output = self.layerS.output

        self.params = self.layerH.params + self.layerS.params

toytoy_set = numpy.asarray([
        [0,0,0,0,
         1,0,0,0,
         0,1,0,0,
         0,0,1,0],
        [0,0,0,0,
         0,1,0,0,
         0,0,1,0,
         0,0,0,1],
        [1,0,0,0,
         0,1,0,0,
         0,0,1,0,
         0,0,0,0],
        [0,1,0,0,
         0,0,1,0,
         0,0,0,1,
         0,0,0,0],
        [0,1,0,0,
         0,1,0,0,
         0,1,0,0,
         0,0,0,0],
        [0,1,0,1,
         0,0,1,0,
         0,0,1,0,
         0,0,0,0],
        [0,0,0,0,
         1,0,0,0,
         1,0,0,0,
         1,0,0,0],
        [0,0,0,0,
         0,1,0,0,
         0,1,0,0,
         0,1,0,0]],dtype='float32')


def train_model():
    
    lr = 0.5
    x = T.matrix()
    x.tag.test_value = numpy.ones((8,28*28),dtype='float32')
    net = Model(x, 28*28, 1000, 100)
    
    #data = toytoy_set
    data = numpy.float32(numpy.random.random((10,28*28)))
    print data.shape
                          
    cost = T.mean((x-net.recon)**2)
    gradients = T.grad(cost, net.params)
    updates = []
    for p,g in zip(net.params, gradients):
        updates.append((p, p - lr*g))

    train_model = theano.function([x],
                                  [cost,net.recon],
                                  updates = updates)

    test_model = theano.function([x],
                                 [cost,net.recon])

    print "Training model"
    for epoch in range(100):
        #t = numpy.float32(data + numpy.random.random(data.shape)*0.2)
        t = data
        err = train_model(t)[0]
        if epoch % 10 == 0:
            print err
    print data
    print test_model(data)[1]

if __name__ == "__main__":
    train_model()
