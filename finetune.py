from baseline import*
import sys
import cPickle as pickle



def main(model_src):
    weights = pickle.load(file(model_src+"/weights.pkl",'r'))
    print [i.shape for i in weights]
    dset = model_src.split("_")[0]

    data = []
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
        data,test = load_faces_xy()
        size = 32,32
        trainx,trainy = data
        testx, testy = test
        

    batch_size = 20
    base_lr = .05
    tau = 20

    lr = T.scalar('lr')
    x = T.matrix('x')
    y = T.matrix('y')

    net = Model(x, 
                weights[0].shape[0], weights[0].shape[1],
                weights[4].shape[1], 
                useCC=len(weights) in [11,9],
                tied=len(weights) in [9])
    #net.import_params(weights)
    
    ftnet = HiddenLayer(net.output, net.layerS.n_out, trainy.shape[1], 
                        T.nnet.softmax, "ft")

    cost = T.mean((y - ftnet.output)**2)
    class_error = 1.0*T.sum(T.neq(T.argmax(y,axis=1), T.argmax(ftnet.output,axis=1))) / y.shape[0]
    params = [i for i in net.params + ftnet.params 
              if ("W" in i.name or 'b' in i.name) and not "'" in i.name]
    print params
    gradients = T.grad(cost, params)
    updates = []
    for p,g in zip(params, gradients):
        updates.append((p, p-lr*g))

    train_model = theano.function([x,y,lr],
                                  [cost, class_error],
                                  updates = updates)

    test_model = theano.function([x, y],
                                 [cost, class_error])

    log = file(model_src+"/ft_log.txt",'a')
    weight_file = file(model_src+"/ft_weights.pkl",'w')


    nminib = trainx.shape[0] / batch_size

    for epoch in range(1000):
        real_lr = (tau * base_lr / (epoch+tau))
        
        i=0
        for batch in range(nminib):
            i+=1
            t = trainx[batch*batch_size:(batch+1)*batch_size]
            y = trainy[batch*batch_size:(batch+1)*batch_size]
            if i % 49 == 0:
                t0 = time.time()
                err = train_model(t,y,real_lr)
                #err, grads = err[:4], err[4:]
                #tt0 = test_model([t[0]])[2][0]
                #print [(abs(k).sum(),p) for k,p in zip(grads,net.params)]
                print err,'(',epoch,real_lr,')',time.time()-t0,'s'
            else:
                err = train_model(t,y,real_lr)
        
        n_test_mini = testx.shape[0] / batch_size
        test_error = 0
        test_cerror = 0
        print "testing:"
        for i in range(n_test_mini+1):
            b = testx[i*batch_size:(i+1)*batch_size]
            y = testy[i*batch_size:(i+1)*batch_size]
            if b.shape[0] == 0: break
            r = test_model(b,y)
            test_error += r[0] * b.shape[0]
            test_cerror += r[1] * b.shape[0]
        test_error /= testx.shape[0]
        test_cerror /= testx.shape[0]
        test_cerror *= 100
        print "epoch",epoch,"error:", err[0],"test error:",test_error,test_cerror,"\b%","with lr",real_lr
        log.write(str(err[0])+" test: "+str(test_error)+' class: '+str(test_cerror)+\
                      '% ( '+str(epoch)+" "+str(real_lr)+' )\n')
        log.flush()
        weight_file.seek(0)
        pickle.dump([i.get_value() for i in params],weight_file,2)
        for img,n in net.export_param_images():
            scipy.misc.imsave(model_src+'/ft_'+n+'.png', img)



if __name__ == "__main__":
    model_src = sys.argv[1]
    print "Finetuning", model_src
    
    main(model_src)
