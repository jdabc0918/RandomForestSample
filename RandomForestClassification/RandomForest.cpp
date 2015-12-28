#include "stdafx.h"

RandomForest::RandomForest(void)    /**< コンストラクタ*/
{
}; 
RandomForest::~RandomForest(void){};    /**< デストラクタ*/
/**
*   特徴ベクトルを計算する
*   @param [in] _in 画像データ
*   @return 特徴ベクトル
**/
RandomForest::FeatureVector RandomForest::calcFeature(Mat _in)
{
    FeatureVector ret;

    //画像情報
    int width = _in.cols;
    int height = _in.rows;
    int sz = width * height;

    //エッジ画像作成
    Mat edgeImg;
    Laplacian(_in, edgeImg, CV_8UC1);

    //輝度画像とエッジ画像でパッチを抽出する
    int psize = PATCH_SIZE; //8
    int px = width / psize;     //横何個
    int py = height / psize;    //縦何個
    for (int iy = 0; iy < py; iy++)
    {
        for (int ix = 0; ix < px; ix++)
        {
            Rect roi(ix * psize, iy * psize, psize, psize);
            Mat pv = _in(roi).clone();
            Mat edge = edgeImg(roi).clone();
            ret.pv.push_back(pv);
            ret.edge.push_back(edge);
        }
    }
    return ret;
}
/**
*   学習サンプルを読み込む
**/
void RandomForest::LoadSamples(void)
{
    //データ情報
    int labelNum = LABEL_NUM;
    int imgNum = 5000;  //5000
    string imgName = "C:/Users/Yosuke/Desktop/CIFAR10/training/%d/%04d.bmp";
    
    //読み込み
    cout << "★サンプルデータ読み込み中...";
    for (int i = 0; i < labelNum; i++)
    {
        for (int j = 0; j < imgNum; j++)
        {
            Data data;  //データ

            //画像読み込み
            char fname[FILENAME_MAX];
            sprintf(fname, imgName.c_str(), i, j);            
            Mat img = imread(fname, 0); //グレースケールで読み込み

            //結果格納
            data.label = i;
            data.feature = this->calcFeature(img);
            this->Samples.push_back(data);
        }
    }
    cout << "終了！" << endl;
}
/**
*   一様乱数を発生させる
*   @return 0.0~1.0の乱数
**/
double RandomForest::rand_uniform(void)
{
    //乱数生成器
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_real_distribution<double> rd(0.0, 1.0);

    //生成
    return rd(engine);
}
/**
*   重複のない乱数リストを生成する
*   @param [in] _size 要素数
*   @param [in] _min  下限
*   @param [in] _max  上限
**/
vector<int> RandomForest::make_rand_array_unique(int _size, int _min, int _max)
{
    if (_min > _max) std::swap(_min, _max);
    const int max_min_diff = (_max - _min + 1);
    if (max_min_diff < _size) throw std::runtime_error("引数が異常です");

    vector<int> tmp;
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_int_distribution<int> distribution(_min, _max);

    const int make_size = (int)(_size * 1.2);

    while ((int)tmp.size() < _size){
        while ((int)tmp.size() < make_size) tmp.push_back(distribution(engine));
        sort(tmp.begin(), tmp.end());
        auto unique_end = unique(tmp.begin(), tmp.end());

        if (_size < distance(tmp.begin(), unique_end)){
            unique_end = next(tmp.begin(), _size);
        }
        tmp.erase(unique_end, tmp.end());
    }

    shuffle(tmp.begin(), tmp.end(), engine);
    return move(tmp);
}
/**
*   学習データをランダムに抽出する
*   @param [in] _num 抽出数
*   @return 抽出データ
**/
vector<RandomForest::Data> RandomForest::getRandomSampleSubset(int _num)
{
    //抽出
    int dataNum = (int)this->Samples.size();
    vector<Data> ret;

    for (int i = 0; i < _num; i++)
    {
        int idx = (int)(dataNum * this->rand_uniform());
        ret.push_back(this->Samples[idx]);
    }
    return ret;
}
/**
*   パラメータをランダムに生成する
*   @return 生成されたパラメータ
**/
RandomForest::SplitParam RandomForest::getRandomParameter(void)
{
    SplitParam ret;

    //ランダムに閾値を生成
    int pv_num = 16, pv_min = 0, pv_max = 255;
    int edge_num = 16, edge_min = 0, edge_max = 255;
    ret.pv_th = this->make_rand_array_unique(pv_num, pv_min, pv_max);
    ret.edge_th = this->make_rand_array_unique(edge_num, edge_min, edge_max);

    //どの要素を使うか(要素数のルート分推奨らしい)
    int fv_length = pv_num + edge_num;
    ret.enable.assign(fv_length, false);    //領域確保
    int enable_num = (int)sqrt(fv_length);
    vector<int> tmp = this->make_rand_array_unique(enable_num, 0, fv_length - 1);
    for (int i = 0; i < enable_num; i++) ret.enable[tmp[i]] = true;

    return ret;    
}
/**
*   データのエントロピーを計算する
*   @param [in] _data 計算対象データ
*   @return エントロピー値
**/
double RandomForest::calcDataEntropy(vector<Data> _data)
{
    //ラベルの存在確率を計算
    vector<double> probs = this->calcProbs(_data);

    //エントロピー計算
    double ret = 0.0;
    for (int i = 0; i < LABEL_NUM; i++)
    {
        ret += probs[i] * log(probs[i]);
    }

    return -ret;
}
/**
*   分岐関数
*   @param [in] _in    入力データ
*   @param [in] _param パラメータ
*   @return true(左)/false(右)
**/
bool RandomForest::splitFunction(Data _in, SplitParam _param)
{
    bool ret = true;
    
    int fv_length = (int)_param.pv_th.size() + (int)_param.edge_th.size();
    for (int i = 0; i < fv_length; i++)
    {
        if (_param.enable[i]) ret &= 
    }
    bool mean_res = (_in.feature.pv_mean < _param.pv_mean_th) ? true : false;
    bool var_res = (_in.feature.pv_var < _param.pv_var_th) ? true : false;

    if ((mean_res && var_res) || (!mean_res && !var_res)) return true;
    else return false;
}
/**
*   葉のラベル存在確率を計算
*   @param [in] _data データリスト
*   @return ラベルの存在確率
**/
vector<double> RandomForest::calcProbs(vector<Data> _data)
{
    vector<double> ret(LABEL_NUM, 0.0);
    int dataNum = (int)_data.size();
    for (int i = 0; i < dataNum; i++)
    {
        int lbl = _data[i].label;
        ret[lbl] += 1.0;
    }
    for (int i = 0; i < LABEL_NUM; i++) ret[i] /= dataNum;
    return ret;
}
/**
*   木を学習する
*   @param [out] _tree 学習対象の木
*   @param [in]  _data 学習に使うサンプルデータ
**/
void RandomForest::LearnTree(Tree *_tree, vector<Data> _data)
{
    //終了判定その１＆その２
    int dataNumTh = 5;  //データ数がこれ以下になったら終了
    int dataNum = (int)_data.size();
    if (dataNum <= dataNumTh)
    {
        _tree->Probs = this->calcProbs(_data);
        cout << "num." << endl;
        return;
    }
    int depthTh = 5;    //深さがこれ以上になったら終了    
    int depth = _tree->getDepth();
    if (depthTh <= depth)
    {
        _tree->Probs = this->calcProbs(_data);
        cout << "depth." << endl;
        return;
    }

    //学習ループ
    SplitParam maxParam;
    vector<Data> maxDataL,maxDataR;
    double maxInfoGain = -1.0;
    //分割前のエントロピー
    double entropy_before = this->calcDataEntropy(_data);
    //パラメータをランダムに生成してループ
    int loop = 100;
    for (int i = 0; i < loop; i++)
    {
        //パラメータ生成
        SplitParam param = this->getRandomParameter();

        //現在のパラメータでデータを分割
        vector<Data> dataL, dataR;
        for (int j = 0; j < dataNum; j++)
        {
            Data d = _data[j];
            if (this->splitFunction(d, param)) dataL.push_back(d);
            else dataR.push_back(d);
        }

        //分割後のエントロピーを計算
        int dataLNum = (int)dataL.size();
        int dataRNum = (int)dataR.size();
        double entropy_after =
            (double)dataLNum / dataNum * this->calcDataEntropy(dataL)
            + (double)dataRNum / dataNum * this->calcDataEntropy(dataR);
        
        //情報利得を計算
        double infoGain = entropy_before - entropy_after;

        //最大値更新
        if (maxInfoGain < infoGain)
        {
            maxInfoGain = infoGain;
            maxParam = param;
            maxDataL = dataL;
            maxDataR = dataR;
        }
    }

    //終了判定その３
    double maxInfoGainTh = 0.01;    //最大情報利得がこれ以下になったら終了
    if (maxInfoGain <= maxInfoGainTh)
    {
        _tree->Probs = this->calcProbs(_data);
        cout << "gain." << endl;
        return;
    }

    //パラメータをセット
    _tree->param = maxParam;

    //子を作成して学習継続
    _tree->left = new Tree;
    _tree->right = new Tree;
    _tree->left->parent = _tree;
    _tree->right->parent = _tree;
    this->LearnTree(_tree->left, maxDataL);
    this->LearnTree(_tree->right, maxDataR);
}
/**
*   森を学習する
**/
void RandomForest::LearnForest()
{
    //サンプル読み込む
    this->LoadSamples();

    //学習ループ
    cout << "☆森の学習中...";
    int treeNum = 100;   //木の数
    for (int i = 0; i < treeNum; i++)
    {
        cout << "i = " << i << endl;
        //ランダムサンプリング
        int subsetSampleNum = 1000;
        vector<Data> data = this->getRandomSampleSubset(subsetSampleNum);

        //サンプリングしたデータで木を学習
        Tree tree;
        this->LearnTree(&tree, data);

        //プッシュ
        this->Forest.push_back(tree);
    }
    cout << "終了！" << endl;
}
/**
*   木の識別結果を返す
*   @param [in] _tree 対象木
*   @param [in] _fv 入力特徴ベクトル
*   @param [out] _res 識別結果
**/
void RandomForest::TestTree(Tree *_tree,FeatureVector _fv,vector<double> &_res)
{
    if (_tree->isLeaf())
    {
        _res =  _tree->Probs;
    }
    else
    {
        Data d;
        d.feature = _fv;
        SplitParam p = _tree->param;
        if (this->splitFunction(d, p)) this->TestTree(_tree->left, _fv, _res);
        else this->TestTree(_tree->right, _fv, _res);
    }
}

/**
*   識別する
*   @param [in] _img 入力画像
*   @param [out] _label 識別結果
**/
void RandomForest::TestForest(Mat _img, int &_label)
{
    //特徴量計算
    FeatureVector fv = this->calcFeature(_img);

    //各木の識別結果を統合
    vector<double> Result(LABEL_NUM, 0.0);
    int treeNum = (int)this->Forest.size();
    for (int i = 0; i < treeNum; i++)
    {
        Tree tree = this->Forest[i];
        vector<double> res;
        this->TestTree(&tree, fv, res);

        for (int j = 0; j < LABEL_NUM; j++) Result[j] += res[j];
    }

    //最大値のラベルを出力
    double probMax = -1.0;
    int labelMax = 0;
    for (int i = 0; i < LABEL_NUM; i++)
    {
        if (probMax < Result[i])
        {
            probMax = Result[i];
            labelMax = i;
        }
    }
    _label = labelMax;
}