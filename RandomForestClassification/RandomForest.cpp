#include "stdafx.h"

RandomForest::RandomForest(void)    /**< �R���X�g���N�^*/
{
}; 
RandomForest::~RandomForest(void){};    /**< �f�X�g���N�^*/
/**
*   �����x�N�g�����v�Z����
*   @param [in] _in �摜�f�[�^
*   @return �����x�N�g��
**/
RandomForest::FeatureVector RandomForest::calcFeature(Mat _in)
{
    FeatureVector ret;

    //�摜���
    int width = _in.cols;
    int height = _in.rows;
    int sz = width * height;

    //�G�b�W�摜�쐬
    Mat edgeImg;
    Laplacian(_in, edgeImg, CV_8UC1);

    //�P�x�摜�ƃG�b�W�摜�Ńp�b�`�𒊏o����
    int psize = PATCH_SIZE; //8
    int px = width / psize;     //������
    int py = height / psize;    //�c����
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
*   �w�K�T���v����ǂݍ���
**/
void RandomForest::LoadSamples(void)
{
    //�f�[�^���
    int labelNum = LABEL_NUM;
    int imgNum = 5000;  //5000
    string imgName = "C:/Users/Yosuke/Desktop/CIFAR10/training/%d/%04d.bmp";
    
    //�ǂݍ���
    cout << "���T���v���f�[�^�ǂݍ��ݒ�...";
    for (int i = 0; i < labelNum; i++)
    {
        for (int j = 0; j < imgNum; j++)
        {
            Data data;  //�f�[�^

            //�摜�ǂݍ���
            char fname[FILENAME_MAX];
            sprintf(fname, imgName.c_str(), i, j);            
            Mat img = imread(fname, 0); //�O���[�X�P�[���œǂݍ���

            //���ʊi�[
            data.label = i;
            data.feature = this->calcFeature(img);
            this->Samples.push_back(data);
        }
    }
    cout << "�I���I" << endl;
}
/**
*   ��l�����𔭐�������
*   @return 0.0~1.0�̗���
**/
double RandomForest::rand_uniform(void)
{
    //����������
    random_device seed_gen;
    default_random_engine engine(seed_gen());
    uniform_real_distribution<double> rd(0.0, 1.0);

    //����
    return rd(engine);
}
/**
*   �d���̂Ȃ��������X�g�𐶐�����
*   @param [in] _size �v�f��
*   @param [in] _min  ����
*   @param [in] _max  ���
**/
vector<int> RandomForest::make_rand_array_unique(int _size, int _min, int _max)
{
    if (_min > _max) std::swap(_min, _max);
    const int max_min_diff = (_max - _min + 1);
    if (max_min_diff < _size) throw std::runtime_error("�������ُ�ł�");

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
*   �w�K�f�[�^�������_���ɒ��o����
*   @param [in] _num ���o��
*   @return ���o�f�[�^
**/
vector<RandomForest::Data> RandomForest::getRandomSampleSubset(int _num)
{
    //���o
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
*   �p�����[�^�������_���ɐ�������
*   @return �������ꂽ�p�����[�^
**/
RandomForest::SplitParam RandomForest::getRandomParameter(void)
{
    SplitParam ret;

    //�����_����臒l�𐶐�
    int pv_num = 16, pv_min = 0, pv_max = 255;
    int edge_num = 16, edge_min = 0, edge_max = 255;
    ret.pv_th = this->make_rand_array_unique(pv_num, pv_min, pv_max);
    ret.edge_th = this->make_rand_array_unique(edge_num, edge_min, edge_max);

    //�ǂ̗v�f���g����(�v�f���̃��[�g�������炵��)
    int fv_length = pv_num + edge_num;
    ret.enable.assign(fv_length, false);    //�̈�m��
    int enable_num = (int)sqrt(fv_length);
    vector<int> tmp = this->make_rand_array_unique(enable_num, 0, fv_length - 1);
    for (int i = 0; i < enable_num; i++) ret.enable[tmp[i]] = true;

    return ret;    
}
/**
*   �f�[�^�̃G���g���s�[���v�Z����
*   @param [in] _data �v�Z�Ώۃf�[�^
*   @return �G���g���s�[�l
**/
double RandomForest::calcDataEntropy(vector<Data> _data)
{
    //���x���̑��݊m�����v�Z
    vector<double> probs = this->calcProbs(_data);

    //�G���g���s�[�v�Z
    double ret = 0.0;
    for (int i = 0; i < LABEL_NUM; i++)
    {
        ret += probs[i] * log(probs[i]);
    }

    return -ret;
}
/**
*   ����֐�
*   @param [in] _in    ���̓f�[�^
*   @param [in] _param �p�����[�^
*   @return true(��)/false(�E)
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
*   �t�̃��x�����݊m�����v�Z
*   @param [in] _data �f�[�^���X�g
*   @return ���x���̑��݊m��
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
*   �؂��w�K����
*   @param [out] _tree �w�K�Ώۂ̖�
*   @param [in]  _data �w�K�Ɏg���T���v���f�[�^
**/
void RandomForest::LearnTree(Tree *_tree, vector<Data> _data)
{
    //�I�����肻�̂P�����̂Q
    int dataNumTh = 5;  //�f�[�^��������ȉ��ɂȂ�����I��
    int dataNum = (int)_data.size();
    if (dataNum <= dataNumTh)
    {
        _tree->Probs = this->calcProbs(_data);
        cout << "num." << endl;
        return;
    }
    int depthTh = 5;    //�[��������ȏ�ɂȂ�����I��    
    int depth = _tree->getDepth();
    if (depthTh <= depth)
    {
        _tree->Probs = this->calcProbs(_data);
        cout << "depth." << endl;
        return;
    }

    //�w�K���[�v
    SplitParam maxParam;
    vector<Data> maxDataL,maxDataR;
    double maxInfoGain = -1.0;
    //�����O�̃G���g���s�[
    double entropy_before = this->calcDataEntropy(_data);
    //�p�����[�^�������_���ɐ������ă��[�v
    int loop = 100;
    for (int i = 0; i < loop; i++)
    {
        //�p�����[�^����
        SplitParam param = this->getRandomParameter();

        //���݂̃p�����[�^�Ńf�[�^�𕪊�
        vector<Data> dataL, dataR;
        for (int j = 0; j < dataNum; j++)
        {
            Data d = _data[j];
            if (this->splitFunction(d, param)) dataL.push_back(d);
            else dataR.push_back(d);
        }

        //������̃G���g���s�[���v�Z
        int dataLNum = (int)dataL.size();
        int dataRNum = (int)dataR.size();
        double entropy_after =
            (double)dataLNum / dataNum * this->calcDataEntropy(dataL)
            + (double)dataRNum / dataNum * this->calcDataEntropy(dataR);
        
        //��񗘓����v�Z
        double infoGain = entropy_before - entropy_after;

        //�ő�l�X�V
        if (maxInfoGain < infoGain)
        {
            maxInfoGain = infoGain;
            maxParam = param;
            maxDataL = dataL;
            maxDataR = dataR;
        }
    }

    //�I�����肻�̂R
    double maxInfoGainTh = 0.01;    //�ő��񗘓�������ȉ��ɂȂ�����I��
    if (maxInfoGain <= maxInfoGainTh)
    {
        _tree->Probs = this->calcProbs(_data);
        cout << "gain." << endl;
        return;
    }

    //�p�����[�^���Z�b�g
    _tree->param = maxParam;

    //�q���쐬���Ċw�K�p��
    _tree->left = new Tree;
    _tree->right = new Tree;
    _tree->left->parent = _tree;
    _tree->right->parent = _tree;
    this->LearnTree(_tree->left, maxDataL);
    this->LearnTree(_tree->right, maxDataR);
}
/**
*   �X���w�K����
**/
void RandomForest::LearnForest()
{
    //�T���v���ǂݍ���
    this->LoadSamples();

    //�w�K���[�v
    cout << "���X�̊w�K��...";
    int treeNum = 100;   //�؂̐�
    for (int i = 0; i < treeNum; i++)
    {
        cout << "i = " << i << endl;
        //�����_���T���v�����O
        int subsetSampleNum = 1000;
        vector<Data> data = this->getRandomSampleSubset(subsetSampleNum);

        //�T���v�����O�����f�[�^�Ŗ؂��w�K
        Tree tree;
        this->LearnTree(&tree, data);

        //�v�b�V��
        this->Forest.push_back(tree);
    }
    cout << "�I���I" << endl;
}
/**
*   �؂̎��ʌ��ʂ�Ԃ�
*   @param [in] _tree �Ώۖ�
*   @param [in] _fv ���͓����x�N�g��
*   @param [out] _res ���ʌ���
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
*   ���ʂ���
*   @param [in] _img ���͉摜
*   @param [out] _label ���ʌ���
**/
void RandomForest::TestForest(Mat _img, int &_label)
{
    //�����ʌv�Z
    FeatureVector fv = this->calcFeature(_img);

    //�e�؂̎��ʌ��ʂ𓝍�
    vector<double> Result(LABEL_NUM, 0.0);
    int treeNum = (int)this->Forest.size();
    for (int i = 0; i < treeNum; i++)
    {
        Tree tree = this->Forest[i];
        vector<double> res;
        this->TestTree(&tree, fv, res);

        for (int j = 0; j < LABEL_NUM; j++) Result[j] += res[j];
    }

    //�ő�l�̃��x�����o��
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