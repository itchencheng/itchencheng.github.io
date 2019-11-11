---
typora-root-url: ..
---

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

x

### 数据(featureMap, weights)都是在哪里分配的?

x

## 以LeNet训练MNIST为例子，讲解Caffe框架整体逻辑

### 1. 下载dataset

```shell
for fname in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
do
    if [ ! -e $fname ]; then
        wget --no-check-certificate http://yann.lecun.com/exdb/mnist/${fname}.gz
        gunzip ${fname}.gz
    fi
done
```

### 2. 将dataset转为LMDB格式

需要写代码，利用LevelDB和LMDB库。

MNIST dataset的详细数据格式，见Lecun网站 http://yann.lecun.com/exdb/mnist/。

![1573487091296](/_posts/2019-08-11-caffe-main.assets/1573487091296.png)

```c++
// Usage: ./convert_mnist_data input_image_file input_label_file output_db_file 
convert_dataset(argv[1], argv[2], argv[3], db_backend);

// 实现
void convert_dataset(const char* image_filename, const char* label_filename,
        const char* db_path, const string& db_backend/*lmdb*/) 
{
    // Open files
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
    // Read the magic and the meta data
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
  	uint32_t cols;
    // 读取文件头信息
    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
    
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
    
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    CHECK_EQ(num_items, num_labels);
    
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    
    scoped_ptr<db::DB> db(db::GetDB(db_backend));
    db->Open(db_path, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    char label;
    char* pixels = new char[rows * cols];
    int count = 0;
    string value;

    Datum datum;
    datum.set_channels(1);
    datum.set_height(rows);
    datum.set_width(cols);
    LOG(INFO) << "A total of " << num_items << " items.";
    LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
    for (int item_id = 0; item_id < num_items; ++item_id) {
        image_file.read(pixels, rows * cols);
        label_file.read(&label, 1);
        datum.set_data(pixels, rows*cols);
        datum.set_label(label);
        string key_str = caffe::format_int(item_id, 8);
        datum.SerializeToString(&value);

        txn->Put(key_str, value);

        if (++count % 1000 == 0) {
        	txn->Commit();
        }
    }
    // write the last batch
    if (count % 1000 != 0) {
    	txn->Commit();
    }
    LOG(INFO) << "Processed " << count << " files.";
    delete[] pixels;
    db->Close();
}
```

### 3. lenet_train_test.prototxt

#### data layer

```protobuf
name: "LeNet"
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
```

训练所用的prototxt，其数据集的描述采用data layer。其data和label由一个lmdb提供。

transform_parm提供了预处理的方式。data_param指明LMDB文件位置，batch_size。

```protobuf
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
```

然后是网络结构的描述

```protobuf
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
  }
  param {
    lr_mult: 2
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
}
```

最后是loss和accuracy

```protobuf
# fc层输出ip2
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "ip2"
  bottom: "label"
  top: "accuracy"
  include {
    phase: TEST
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip2"
  bottom: "label"
  top: "loss"
}
```

我们来看一下DataLayer的实现：



### 4. lenet_solver.protoxt

```protobuf
# The train/test net protocol buffer definition
net: "examples/mnist/lenet_train_test.prototxt"
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100
# Carry out testing every 500 training iterations.
test_interval: 500
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0005
# The learning rate policy
lr_policy: "inv"
gamma: 0.0001
power: 0.75
# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 10000
# snapshot intermediate results
snapshot: 5000
snapshot_prefix: "examples/mnist/lenet"
# solver mode: CPU or GPU
solver_mode: GPU

```

