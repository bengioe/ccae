# mnist:
#   AE 400 400 50 7.11 (61 epochs)
# CCAE 400 400 50 8.41 (60 epochs)
#   AE 400 400 10 14.83 (60 epochs)
# CCAE 400 400 10 17.33 (60)
#   AE 400 400 20 9.3   (60)
# CCAE 400 400 20 12.4  (60)

import numpy
import theano
import theano.tensor as T
from math import*

import cPickle as pickle
import scipy.ndimage as ndimg

import os
import os.path
import time
import scipy

from geom_dataset import*

do_pygame = 0

if do_pygame:
    import pygame


rng = numpy.random.RandomState(42)
if do_pygame:
    pygame.init()
    screen = pygame.display.set_mode((1000,800))

#theano.config.compute_test_value = 'warn'

sqrt = numpy.sqrt
xpower = 1

"""
Essayer juste une couche
Remplacer sigmoide par rien

mu et Wcc tied
pour D (10,200) groupes de 20, sans les apprendres potentiellement

"""

class HiddenLayer:
    def __init__(self,x,n_in,n_out,activation = T.tanh,name=''):
        self.name = name
        self.x = x
        self.n_in = n_in
        self.n_out = n_out
        k = sqrt(6.0/(n_in+n_out)) * 10
        W_vals = numpy.asarray(rng.uniform(-k,k,size=(n_in,n_out)))**xpower
        #W_vals = numpy.asarray(rng.normal(0,1,size=(n_in,n_out)))
        #W_vals /= W_vals.max()
        print (n_in,n_out),k
        self.W = theano.shared(value=W_vals,name=name+'_W')
        self.b = theano.shared(0.25*numpy.ones((n_out,)),name=name+'_b')
        self.output = activation(T.dot(self.x,self.W)+self.b)
        self.params = [self.W,self.b]
        self.L2 = T.sum(self.W**2)
        
    def recon_from(self, xp, activation=T.nnet.sigmoid, tied=True):
        #xp = theano.printing.Print('xp',('__str__','max'))(xp)
        self.xp = xp
        
        if tied:
            self.Wp = self.W.T
        else:
            k = sqrt(6.0/(self.n_in+self.n_out))
            W_vals = numpy.asarray(rng.uniform(-k,k,size=(self.n_out,self.n_in)))**xpower
            #W_vals /= W_vals.max() 
            self.Wp = theano.shared(value=W_vals, name=self.name+"_W'")
            self.params.append(self.Wp)
            self.L2 += T.sum(self.Wp**2)
        self.bp = theano.shared(numpy.zeros((self.n_in,)), name=self.name+"_b'")
        self.recon = activation(T.dot(xp,self.Wp) + self.bp)
        self.params.append(self.bp)

class CCLayer:
    def __init__(self, x, n_in, n_out, name='',version=5):
        self.name = name
        self.x = x
        self.n_in = n_in
        self.n_out = n_out
        self.version = version
        if version == 1 or version == 2 or version == 4:
            k = sqrt(6.0e-9/(n_in+n_out))
            W_vals = numpy.asarray(rng.uniform(-k,k,size=(n_in,n_out)))
            #W_vals /= W_vals.max()
            print "CC",(n_in,n_out),k,W_vals.min(),W_vals.max()
            self.W = theano.shared(value=W_vals,name=name+'_W')
            self.b = theano.shared(numpy.zeros((n_out,)),name=name+'_b')
            b = self.b#theano.printing.Print('b',('min','max'))(self.b)
            W = self.W#theano.printing.Print('W',('min','max'))(self.W)
            self.output = T.tanh(T.dot(self.x,W)+b)
            self.params = [self.W,self.b]
            self.L2 = T.sum(self.W**2)
        elif version == 5:
            k = sqrt(6.0e-9/(n_in+n_out))
            W_vals = numpy.asarray(rng.uniform(-k,k,size=(n_in,n_out)))
            print "CC",(n_in,n_out),k,W_vals.min(),W_vals.max()
            self.W = theano.shared(value=W_vals,name='Wcc')
            W = self.W#theano.printing.Print('W',('min','max'))(self.W)
            self.output = T.dot(self.x,W)
            self.params = [self.W]
            self.L2 = T.sum(self.W**2)
        elif version == 3:
            self.mu = theano.shared(
                rng.uniform(-1,1,size=(self.n_in,self.n_out)),
                name = 'mu')
            self.rho = theano.shared(
                rng.uniform(0,1,size=(self.n_out,)),
                name = 'rho')
            # K \in (bs,nin,nout)
            K = x.dimshuffle(0,1,'x') - self.mu.dimshuffle('x',0,1)
            self.output = T.exp(-K.sum(axis=1)*self.rho.dimshuffle('x',0))
            self.params = [self.mu,self.rho]
            self.L2 = 0
            

    def recon_from(self, s, tied):
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

        elif self.version in [2,3,4]:
            # s = (bs, n_out)
            if self.version == 2 or self.version==4:
                self.mu = theano.shared(
                    rng.uniform(-1,1,size=(self.n_in,self.n_out)),
                    name = 'mu')
                mu = self.mu
            else:
                self.mup = theano.shared(
                    rng.uniform(-1,1,size=(self.n_in,self.n_out)),
                    name = 'mu\'')
                mu = self.mup
            D = numpy.zeros((self.n_in,self.n_out,self.n_out))+0.05
            #D = numpy.zeros((self.n_out,self.n_out))+0.05
            for i in range(self.n_in):
                numpy.fill_diagonal(D[i], 1)
            #numpy.fill_diagonal(D, 1)
            self.D = theano.shared(D, name='D')
            self.pow = theano.shared(
                rng.uniform(0,1,size=(self.n_in,)),
                name = 'pw')
            self.params += [self.D, mu]#, self.pow]
            K = s.dimshuffle('x',0,1) - mu.dimshuffle(0,'x',1)
            #K = theano.printing.Print('K')(K)
            #L,_ = theano.map(lambda x:T.dot(x,self.D), [K])
            L,_ = theano.map(T.dot, [K,self.D])
            Z = T.mul(L, K).mean(axis=2)
            #Z = theano.printing.Print("Z")(Z)
            V = T.exp(-Z)
            R = V.T
            self.recon = R

        elif self.version == 5:
            # s = (bs, n_out)
            # mu = (n_in, n_out)
            self.mu = theano.shared(
                rng.uniform(-1,1,size=(self.n_in,self.n_out)),
                name = 'mu')
            mu = self.mu
            if 0:
                D = numpy.zeros((self.n_in,self.n_out)) + 0.0001
                for i in range(self.n_in):
                    D[i][numpy.random.randint(0,self.n_out)] = 1
            else:
                D = rng.uniform(0,1,size=(self.n_in,self.n_out))
            self.D = theano.shared(D, name='D')
            self.params += [self.D, mu]
            # K = (bs, n_in, n_out)
            K = (s.dimshuffle(0,'x',1) - mu.dimshuffle('x',0,1))**2
            L = K * abs(self.D).dimshuffle('x',0,1)
            Z = L.mean(axis=2)
            R = T.exp(-Z)
            self.recon = R
        """
        r = []
        for i in range(self.n_in):
            k = (s-self.mu[i].reshape((1,self.n_out)))
            l = T.dot(k, self.D[i])
            # v = T.exp(-T.dot(l, k.T)).diagonal()
            # but the dot is expensive for nothing  we're only taking the diagonal
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
    def __init__(self, x, n_in, n_hidden, n_cch, useCC, tied=True):
        self.layerH = HiddenLayer(x, n_in, n_hidden,T.nnet.sigmoid,'h1')
        self.layerH2 = HiddenLayer(self.layerH.output, n_hidden, n_hidden, T.nnet.sigmoid,'h2')
        if useCC:
            print "Using CC"
            self.layerS = CCLayer(self.layerH2.output, n_hidden, n_cch,'cc')
        else: 
            self.layerS = HiddenLayer(self.layerH2.output, n_hidden, n_cch,T.nnet.sigmoid,'h3')


        self.layerS.recon_from(self.layerS.output,tied=tied)
        self.layerH2.recon_from(self.layerS.recon,tied=tied)
        self.layerH.recon_from(self.layerH2.recon,tied=tied)
        self.recon = self.layerH.recon
        self.output = self.layerS.output
        G = T.as_tensor_variable(0)
        if useCC:
            #hrs_grad = T.grad(self.layerS.recon.sum(), self.layerS.output)
            #self.eq_cost = abs(hrs_grad.sum())# - (hrs_grad**2).sum()
            hr = self.layerS.recon # (bs, nh)
            mu = self.layerS.mu    # (nh, nc)
            D = self.layerS.D      # (nh, nc)
            s = self.layerS.output # (bs, nc)
            # en vector on veut: 
            #G = hr.dimshuffle(0,'x') * D * (s.dimshuffle('x',0) - mu)
            # G in (nh, nc)
            # en matrice par contre:
            G = -2 * hr.dimshuffle(0,1,'x') * D.dimshuffle('x',0,1) * \
                (s.dimshuffle(0,'x',1) - mu.dimshuffle('x',0,1))
        # G in (bs, nh, nc)
        self.G = G
        
        self.L2 = self.layerS.L2 + self.layerH.L2
        self.params = self.layerH.params  + self.layerS.params + self.layerH2.params

    def export_params(self):
        return [i.get_value() for i in self.params]

    def import_params(self,ps):
        for p,v in zip(self.params,ps):
            print p.name,v.shape
            p.set_value(v)
    
    def export_param_images(self):
        ars = []
        for p in self.params:
            v = p.get_value()
            if 'W' in p.name and int(sqrt(v.shape[0])) == sqrt(v.shape[0]):
                size = int(sqrt(v.shape[0]))
                v = v.T
                v -= v.min(axis=1).reshape((v.shape[0],1))
                v /= v.max(axis=1).reshape((v.shape[0],1))
                v = numpy.uint8(255*v)
                print v.shape,p.name
                nvert = v.shape[0] / (1000 / size)
                if nvert == 0 : nvert = 1
                nhor = v.shape[0] / nvert + 1
                print nvert,nhor
                if nvert * nhor != v.shape[0]:
                    v = numpy.vstack((v,numpy.zeros((nvert*nhor-v.shape[0],v.shape[1]))))
                print v.shape
                v = v.reshape((nvert,nhor,size,size))
                v = numpy.hstack(numpy.hstack(v))
                print v.shape
            elif v.ndim == 2:
                v -= v.min()
                v /= v.max()
                v = numpy.uint8(255*v)
            elif 'b' in p.name:
                continue
            else:
                print v.shape,"dont know how to deal with!",p.name
            ars.append((v,p.name))
        return ars


    def image_test_batch(self,a,b,sz):
        c = numpy.hstack(
            numpy.hstack((a.reshape((a.shape[0], sz[0], sz[1])),
                          b.reshape((b.shape[0], sz[0], sz[1])))))
        c -= c.min()
        c /= c.max()
        c = numpy.uint8(c*255)
        
        return c
                
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
    data = pickle.load(file(path,'r'))
    print len(data)
    return {"train":(data[0][0],data[0][1]),
            "test":(data[1][0],data[1][1])}

def load_faces():
    path = "FacsEmots.pkl"
    data = pickle.load(file(path,'r'))
    a,b = numpy.array(data[0][0]),numpy.array(data[1][0])
    a = numpy.float32(a) / 255.0
    b = numpy.float32(b) / 255.0
    return a,b

def load_faces_xy(shuffle=True):
    path = "FacsEmots.pkl"
    data = pickle.load(file(path,'r'))
    train = [(data[0][0][i],data[0][2][i])
             for i in range(len(data[0][0])) if not -1 in data[0][2][i]]
    if shuffle:
        numpy.random.shuffle(train)
    imgtrain,ytrain = zip(*train)
    print len(ytrain),len(imgtrain)
    imgtest = numpy.float32(data[1][0])
    ytest = numpy.float32(data[1][2])

    imgtrain = numpy.float32(imgtrain) / 255.0
    imgtest = numpy.float32(imgtest) / 255.0
    return [imgtrain,numpy.float32(ytrain)],[imgtest,ytest]

def np2img(n, size=(28,28), scale =1):
    n = n.reshape(*size).T.reshape(list(size)+[1])*numpy.ones((1,1,3))*255
    n = pygame.surfarray.make_surface(n)
    if scale!=1:
        n = pygame.transform.scale(n, [i*scale for i in size])
    return n

def train_model():
    dset = 'faces'

    data = toytoy_set
    #data = numpy.float32(numpy.random.random((10,28*28)))
    if dset == 'mnist':
        # epoch 352, nll 68.86
        mnist = load_mnist()
        data,_ = mnist['train']
        test,_ = mnist['test']
        numpy.random.shuffle(data)
        size = 28,28
    elif dset is 'geom':
        data = create_geom_dataset(16)
        numpy.random.shuffle(data)
        test = data[:-50]
        data = data[:-50]
        cont_test = data[:16]
        size = 16,16
    elif dset == 'faces':
        #epoch 492 nll 599.20
        data,test = load_faces()
        numpy.random.shuffle(data)
        size = 32,32

    print data.shape, test.shape
    
    batch_size = 20
    #GEOM:  lr 0.5, (250,250,20), bs=20, tau=50
    #MNIST: lr 0.1, (500,500,40), bs=20, tau=50
    #FACES: lr 0.1, (500,500,40), bs=20, tau=50
    base_lr = .0025
    tau = 200
    n_hidden = 400
    n_cc = 50
    use_cc = True
    redux = 0.000095

    exp_name = "%s_%d_%d_%s_%f"%(dset, n_hidden, n_cc,
                                 "cc" if use_cc else "mlp",
                                 redux)


    if not os.path.exists(exp_name):
        os.mkdir(exp_name)

    log = file(exp_name+"/log.txt",'a')
    weight_file = file(exp_name+"/weights.pkl",'w')

    print data
    lr = T.scalar()
    x = T.matrix()
    x.tag.test_value = numpy.ones((8,28*28),dtype='float32')
    net = Model(x, data.shape[1], n_hidden, n_cc, use_cc)
    print net.params
                   
    
    #cost = T.sum((x-net.recon)**2) / x.shape[0]
    cost = -T.sum(x*T.log(net.recon)+(1-x)*T.log(1-net.recon)) / x.shape[0]#.prod()
    rcost = cost + redux*(abs(net.G.sum()) - abs(net.G).sum())# + net.eq_cost * 0.1#+ 0.00001 * net.L2
    gradients = T.grad(rcost, net.params)
    updates = []
    for p,g in zip(net.params, gradients):
        updates.append((p, p - lr*g))

    train_model = theano.function([x,lr],
                                  [cost,rcost,abs(net.G).sum(),abs(net.G.sum())]+gradients,
                                  updates = updates)

    test_model = theano.function([x],
                                 [rcost,cost,net.recon, (net.G**2).sum()])
    get_repr = theano.function([x],
                               [net.output,net.layerH.output])

    print "Training model"
    nminib = data.shape[0] / batch_size
    i = 0

    noisePretrain = 0
    pretrain_lr = 0.005
    NK = 500000
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
                print [(abs(k).sum(),p) for k,p in zip(grads,net.params)]
                print linecat(np2txt(test_model([t[0]])[2][0]),
                              np2txt(t[0]))
                print err,'(',epoch,real_lr,')',time.time()-t0,'s'
                print net.layerH.bp.get_value().mean()
            else:
                err = train_model(t,real_lr)


    for epoch in range(1000):
        #t = numpy.float32(data + numpy.random.random(data.shape)*0.2)
        real_lr = (tau * base_lr / (epoch+tau))
        
        for batch in range(nminib):
            i+=1
            t = data[batch*batch_size:(batch+1)*batch_size]
            #t = t + numpy.float32(numpy.random.normal(0,1,t.shape)*0.1)
            if True and False:
                t = displace(t,(28,28),8)
            if i % 49 ==0:
                t0 = time.time()
                err = train_model(t,real_lr)
                err, grads = err[:4], err[4:]
                tt0 = test_model([t[0]])[2][0]
                print [(abs(k).sum(),p) for k,p in zip(grads,net.params)]
                if 0:
                    print linecat(
                        linecat(np2txt(test_model([data[0]])[2][0]),
                                np2txt(data[0])),
                        linecat(np2txt(tt0),
                                np2txt(t[0])))
                print err,'(',epoch,real_lr,')',time.time()-t0,'s'
                #print get_repr(cont_test)
                
                t = abs(t)
                if do_pygame:
                    ra = numpy.random.randint(0,nminib)
                    t = data[ra*batch_size:(ra+1)*batch_size]
                    r = test_model(t)
                    asd = r[3]
                    r = r[2]
                    print "ASD",asd
                    for k in range(10):
                        img = np2img(r[k],size=size,scale=2)
                        screen.blit(img, (size[0]*k*2,0))
                        img = np2img(t[k],size=size,scale=2)
                        screen.blit(img, (size[0]*k*2,size[1]*2))
                    W = net.layerH.W.get_value().T
                    k = 3
                    xpos = 0
                    scale = 1
                    for j in grads[0].T[:25*6]:#W:
                        j = j.reshape(size)
                        j -= j.min()
                        j /= j.max()
                        img = np2img(j,size=size,scale=scale)
                        screen.blit(img, ((size[0]+1)*xpos*scale,(size[1]+1)*scale*k+50))
                        xpos += 1
                        if xpos > 24:
                            k+=1
                            xpos = 0
                    for j in W[:25*6]:
                        j = j.reshape(size)
                        j -= j.min()
                        j /= j.max()
                        img = np2img(j,size=size,scale=scale)
                        screen.blit(img, ((size[0]+1)*xpos*scale,(size[1]+1)*scale*k+50))
                        xpos += 1
                        if xpos > 24:
                            k+=1
                            xpos = 0
                        
                    pygame.display.flip()

            else:
                err = train_model(t,real_lr)
                #print  [(abs(k).sum(),p) for k,p in zip(err[2:],net.params)]
        
        n_test_mini = test.shape[0] / batch_size
        test_error = 0
        for i in range(n_test_mini+1):
            b = test[i*batch_size:(i+1)*batch_size]
            if b.shape[0] == 0: break
            r = test_model(b)
            test_error += r[0] * b.shape[0]
        test_error /= test.shape[0]

        print "epoch",epoch,"error:", err[0],"test error:",test_error,"with lr",real_lr
        log.write(str(err[0])+" test: "+str(test_error)+' ( '+str(epoch)+" "+str(real_lr)+' )\n')
        log.flush()
        weight_file.seek(0)
        pickle.dump(net.export_params(),weight_file,2)
        for img,n in net.export_param_images():
            scipy.misc.imsave(exp_name+'/'+n+'.png', img)
        test_bunch = test[numpy.random.randint(0, test.shape[0], 20)]
        img = net.image_test_batch(test_bunch, test_model(test_bunch)[2], size)
        scipy.misc.imsave(exp_name+'/test.png', img)
    print data
    print numpy.round(test_model(data[:3])[1])

if __name__ == "__main__":
    try:
        train_model()
    finally:
        if do_pygame:
            pygame.quit()
