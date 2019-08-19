## Caffe主程序代码解读

./tool/caffe.cpp

### train

一个典型的solver.protoxt

```protobuf
net: "models/bvlc_alexnet/train_val.prototxt"
test_iter: 1000
test_interval: 1000
base_lr: 0.01
lr_policy: "step"
gamma: 0.1
stepsize: 100000
display: 20
max_iter: 450000
momentum: 0.9
weight_decay: 0.0005
snapshot: 10000
snapshot_prefix: "models/bvlc_alexnet/caffe_alexnet_train"
solver_mode: GPU
```

读取solver.prototxt到solver_param结构。

```c++
// 读取solver.protoxt
caffe::SolverParameter solver_param;
caffe::ReadSolverParamsFromTextFileOrDie(FLAGS_solver, &solver_param);
```

由solver_param创建Solver。

```c++
shared_ptr<caffe::Solver<float> >
  solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
```

注：这里有个SolverRegistry，之前Layers有个LayerRegistry。

### 关于XxxRegistry和XxxRegister

XxxRegistry和XxxRegister是两个类，用于管理同一大类，不同小类。

#### 以SolverRegistry为例

SolverRegistry是一个类，定义和实现都在./caffe/include/solver_factory.hpp

SolverRegistry的成员函数全是static函数，即无需构造对象，类名即可调用。

以SGD函数的注册为例：

在sgd_solver.cpp中，前面大部分是SGDSolver类的实现，最后是：

```c++
INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);
```

INSTANTIATE_CLASS的实现位于common.hpp

用于讲模板定义的类，手动实例化。

```c++
// Instantiate a class with float and double specifications.
//==================== 入口 ============================================
#define INSTANTIATE_CLASS(classname) \
  char gInstantiationGuard##classname; \
  template class classname<float>; \
  template class classname<double>
```

REGISTER_SOLVER_CLASS的具体实现位于solver_factory.hpp。

相当于定义了两个static的静态全局SolverRegister对象。SolverRegister对象只有一个构造函数：其功能时调用SolverRegistry的静态函数：AddCreator(const string& type, Creator creator) 。会去创建一个局部static变量（map<string,  函数指针>），在第一次调用这个函数的时候，会new这样一个map，用来存储<name, solver_creator>。向其中加入一个映射。

```c++
template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,
      Solver<Dtype>* (*creator)(const SolverParameter&)) {
    // LOG(INFO) << "Registering solver type: " << type;
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};

#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \

//==================== 入口 ============================================
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)
```

 总结一下：

基础工具（两个类）：

**SolverRegistry类**，提供static成员函数

1. 使用类名创建一个映射表。并提供向内添加的接口函数。
2. 调用映射表的中构造函数，根据参数，new一个solver。

**SolverRegister类**，用来构造全局静态对象。只需提供一个构造函数，能将构造函数入参，加入map中。

对于每一个待添加的Solver，调用宏：

1. 定义一个函数，能new对应的Solver。
2. 讲名字和函数作为参数，构造SolverRegister的对象（float和double各一个）。

现在，再来看一下使用方法：

```c++
shared_ptr<caffe::Solver<float> >
  solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
```

使用SolverRegistry中的static函数，根据solver_param中的solver类型字符串，从映射表（注册表项）中查找到相应的类型，并执行构造函数。



### SGDSolver根据solver_param构造时做了什么？

SGDSolver构造函数

```c++
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
```

基类Solver构造函数

```c++
template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_(), callbacks_(), requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed() + Caffe::solver_rank());
  }
  // Scaffolding code
  InitTrainNet();
  InitTestNets();
  if (Caffe::root_solver()) {
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

```

PreSolve()

```c++
template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}
```

构造SGD Solver主要做的事情是调用Solver类的构造函数，会初始化TrainNet和TestNet。然后执行PreSolve。

这里关于Net的构建非常重要：

Net类的构造函数处理网络的构造，不论Train和Test，都能处理。其构造函数就是调用了一个巨长的、根据net.prototxt为参数的Init函数。

## Net

### Init进行网络的构造，是Caffe的精华

```c++
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  bottom_need_backward_.resize(param.layer_size());
```



