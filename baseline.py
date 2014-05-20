import numpy
import theano
import theano.tensor as T
from math import*
import scipy.ndimage as ndimg

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
        self.L2 = T.sum(self.W**2)
        
    def recon_from(self, xp, activation=T.nnet.sigmoid, tied=False):
        #xp = theano.printing.Print('xp',('__str__','max'))(xp)
        self.xp = xp
        
        if tied:
            self.Wp = self.W.T
        else:
            k = sqrt(6.0/(self.n_in+self.n_out))
            W_vals = numpy.asarray(rng.uniform(-k,k,size=(self.n_out,self.n_in)))
            self.Wp = theano.shared(value=W_vals, name="W'")
            self.params.append(self.Wp)
            self.L2 += T.sum(self.Wp**2)
        self.bp = theano.shared(numpy.zeros((self.n_in,)), name="b'")
        self.recon = activation(T.dot(xp,self.Wp) + self.bp)
        self.params.append(self.bp)

class CCLayer:
    def __init__(self, x, n_in, n_out, version=2):
        self.n_in = n_in
        self.n_out = n_out
        self.version = version
        if version == 1 or version == 2:
            self.x = x
            self.n_in = n_in
            self.n_out = n_out
            k = sqrt(6.0/(n_in+n_out))
            W_vals = numpy.asarray(rng.uniform(-k,k,size=(n_in,n_out)))
            print (n_in,n_out)
            self.W = theano.shared(value=W_vals,name='Wcc')
            self.b = theano.shared(0.25*numpy.ones((n_out,)),name='bcc')
            self.output = T.tanh(T.dot(self.x,self.W)+self.b)
            self.params = [self.W,self.b]
            self.L2 = T.sum(self.W**2)

    def recon_from(self, s):
        if self.version == 1:
            # s = (bs, n_out)
            self.mu = theano.shared(
                rng.uniform(-0.1,0.1,size=(self.n_in,self.n_out)),
                name = 'mu')
            #D = numpy.zeros((self.n_in,self.n_out,self.n_out))+0.05
            D = numpy.zeros((self.n_out,self.n_out))+0.05
            #for i in range(self.n_in):
            #    numpy.fill_diagonal(D[i], 1)
            numpy.fill_diagonal(D, 1)
            self.D = theano.shared(D, name='D')
            self.params += [self.mu, self.D]
            K = s.dimshuffle('x',0,1) - self.mu.dimshuffle(0,'x',1)
            L,_ = theano.map(lambda x:T.dot(x,self.D), [K])
            V = T.exp(-T.mul(L, K).sum(axis=2))
            R = V.T
            self.recon = R
        elif self.version == 2:
            # s = (bs, n_out)
            self.mu = theano.shared(
                rng.uniform(-1,1,size=(self.n_in,self.n_out)),
                name = 'mu')
            D = numpy.zeros((self.n_in,self.n_out,self.n_out))+0.05
            #D = numpy.zeros((self.n_out,self.n_out))+0.05
            for i in range(self.n_in):
                numpy.fill_diagonal(D[i], 1)
            #numpy.fill_diagonal(D, 1)
            self.D = theano.shared(D, name='D')
            self.pow = theano.shared(
                rng.uniform(0,1,size=(self.n_in,)),
                name = 'pw')
            self.params += [self.mu, self.D]#, self.pow]
            K = s.dimshuffle('x',0,1) - self.mu.dimshuffle(0,'x',1)
            #K = theano.printing.Print('K')(K)
            #L,_ = theano.map(lambda x:T.dot(x,self.D), [K])
            L,_ = theano.map(T.dot, [K,self.D])
            Z = T.mul(L, K).mean(axis=2)
            #Z = theano.printing.Print("Z")(Z)
            V = T.exp(-Z)
            R = V.T
            self.recon = R
        elif self.version == 3:
            self.mu = theano.shared(
                rng.uniform(0,1,size=(self.n_in,self.n_out)),
                name = 'mu')
            
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
        """
        K = s.dimshuffle('x',0,1) - self.mu.dimshuffle(0,'x',1)
        #numpy.sum(a[:,:,:,numpy.newaxis]*b[:,numpy.newaxis,:,:],axis=-2)
        L = T.sum(K.dimshuffle(0,1,2,'x')*self.D.dimshuffle(0,'x',1,2),axis=-2)
        V = T.exp(-T.mul(L, K).sum(axis=2))
        R = V.T
        self.recon = R
        """
        #"""

class Model:
    def __init__(self, x, n_in, n_hidden, n_cch, useCC):
        self.layerH = HiddenLayer(x, n_in, n_hidden,T.nnet.sigmoid)
        if useCC:
            print "Using CC"
            self.layerS = CCLayer(self.layerH.output, n_hidden, n_cch)
        else:
            self.layerS = HiddenLayer(self.layerH.output, n_hidden, n_cch,T.nnet.sigmoid)
        self.layerS.recon_from(self.layerS.output)
        self.layerH.recon_from(self.layerS.recon)
        self.recon = self.layerH.recon
        self.output = self.layerS.output
        
        self.L2 = self.layerS.L2 + self.layerH.L2
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

def displace(n,size,amt):
    n = n.reshape(n.shape[0],size[0],size[1]).copy()
    for i in range(len(n)):
        ndimg.interpolation.shift(n[i],[numpy.random.random()*amt-amt/2,
                                        numpy.random.random()*amt-amt/2], n[i])
    return n.reshape(n.shape[0],size[0]*size[1])

def np2txt(n,size=(28,28)):
    n = n.reshape(*size)
    s = ""
    for i in range(size[0]):
        for j in range(size[1]):
            if n[i,j] > 0.9:
                s += "@"
            elif n[i,j] > 0.75:
                s += "#"
            elif n[i,j] > 0.675:
                s += "0"
            elif n[i,j] > 0.5:
                s += "O"
            elif n[i,j] > 0.375:
                s += "i"
            elif n[i,j] > 0.25:
                s += "-"
            elif n[i,j] > 0.1:
                s += "."
            else:
                s += " "
        s += "\n"
    return s

def linecat(a,b,sep="|"):
    a = a.splitlines()
    b = b.splitlines()
    return "\n".join(i+sep+j for i,j in zip(a,b))
    

def load_mnist(path='./mnist.pkl'):
    import cPickle as pickle
    data = pickle.load(file(path,'r'))
    print len(data)
    return {"train":(data[0][0],data[0][1])}
    


def train_model():
    
    data = toytoy_set
    #data = numpy.float32(numpy.random.random((10,28*28)))
    data,labels = load_mnist()['train']#[:5000]
    numpy.random.shuffle(data)
    print data.shape
    batch_size = 50
    base_lr = .03
    tau = 5
    size = 28,28

    lr = T.scalar()
    x = T.matrix()
    x.tag.test_value = numpy.ones((8,28*28),dtype='float32')
    net = Model(x, data.shape[1], 400, 20, False or True)
    print net.params
                          
    #cost = T.sum((x-net.recon)**2) / batch_size
    cost = -T.sum(x*T.log(net.recon)+(1-x)*T.log(1-net.recon)) / batch_size
    rcost = cost #+ 0.00001 * net.L2
    gradients = T.grad(rcost, net.params)
    updates = []
    for p,g in zip(net.params, gradients):
        updates.append((p, p - lr*g))

    train_model = theano.function([x,lr],
                                  [rcost,cost]+gradients,
                                  updates = updates)

    test_model = theano.function([x],
                                 [rcost,cost,net.recon])

    print "Training model"
    nminib = data.shape[0] / batch_size
    import time
    i = 0

    noisePretrain = 0
    pretrain_lr = 0.005
    NK = 500
    for epoch in range(noisePretrain):
        real_lr = (tau * pretrain_lr / (epoch+tau))
        
        for batch in range(nminib):
            i+=1
            t = numpy.float32(numpy.random.rand(batch_size / 2, size[0], size[1]))*0.4# * (0.2 + (batch % 10) / 10.)
            t = numpy.fft.fft2(t)
            t = ndimg.interpolation.affine_transform(numpy.abs(t),numpy.random.rand(3,3))
            t = numpy.float32(numpy.abs(numpy.fft.ifft2(t))).reshape((batch_size / 2, size[0]* size[1]))
            if i*batch_size > NK:
                i = 0
                t0 = time.time()
                NK = int(numpy.random.random() * 50) + 500
                err = train_model(t,real_lr)
                err, grads = err[:2], err[2:]
                print [((k**2).sum(),p) for k,p in zip(grads,net.params)]
                print linecat(np2txt(test_model([t[0]])[2][0]),
                              np2txt(t[0]))
                print err,'(',epoch,real_lr,')',time.time()-t0,'s'
                print net.layerH.bp.get_value().mean()
            else:
                err = train_model(t,real_lr)


    for epoch in range(10000):
        #t = numpy.float32(data + numpy.random.random(data.shape)*0.2)
        real_lr = (tau * base_lr / (epoch+tau))
        
        for batch in range(nminib):
            i+=1
            t = data[batch*batch_size:(batch+1)*batch_size]
            if True and False:
                t = displace(t,(28,28),8)
            if i*batch_size % 500 ==0:
                t0 = time.time()
                err = train_model(t,real_lr)
                err, grads = err[:2], err[2:]
                print [((k**2).sum(),p) for k,p in zip(grads,net.params)]
                print linecat(
                    linecat(np2txt(test_model([data[0]])[2][0]),
                            np2txt(data[0])),
                    linecat(np2txt(test_model([t[0]])[2][0]),
                            np2txt(t[0])))
                print err,'(',epoch,real_lr,')',time.time()-t0,'s'
                print net.layerH.bp.get_value().mean()
            else:
                err = train_model(t,real_lr)
        print "epoch",epoch,"error:", err, "with lr",real_lr
    print data
    print numpy.round(test_model(data[:3])[1])

if __name__ == "__main__":
    train_model()
