#pragma once

#include "stdafx.h"

#define LABEL_NUM 10
#define PATCH_SIZE 8

class RandomForest
{
private:
    /**
    *   @struct �����x�N�g��
    **/
    typedef struct FeatureVector
    {
        vector<Mat>  pv;   /**< �P�x�p�b�`*/
        vector<Mat>  edge; /**< �G�b�W�p�b�`*/
    };

    /**
    *   @struct �T���v���f�[�^���
    **/
    typedef struct Data
    {
        int label;              /**< ���x���l*/
        FeatureVector feature;  /**< �����x�N�g��*/
    };

    /**
    *   @struct ����֐��p�����[�^
    **/
    typedef struct SplitParam
    {
        vector<bool> enable;        /**< �g�����ǂ����̃e�[�u��*/
        vector<int>  pv_th;      /**< �P�x臒l*/
        vector<int>  edge_th;    /**< �G�b�W臒l*/
    };

    /**
    *   @struct ����؍\����
    **/
    typedef struct Tree
    {
        SplitParam param;   /**< ����֐��p�����[�^*/
        Tree *left;         /**< �q�m�[�h(��)�ւ̃|�C���^*/
        Tree *right;        /**< �q�m�[�h(�E)�ւ̃|�C���^*/
        Tree *parent;       /**< �e�m�[�h�ւ̃|�C���^*/
        vector<double> Probs;   /**< �e���x���̑��݊m��(�t����)*/
        Tree()
        {
            left = NULL;
            right = NULL;
            parent = NULL;
            Probs.assign(LABEL_NUM, 0.0);
        }
        ~Tree(){}
        /**
        *   ���M���e�m�[�h���ǂ�����Ԃ�
        *   @return true(�e)/false(����ȊO)
        **/
        bool isRoot(void)
        {
            return (parent == NULL) ? true : false;
        }
        /**
        *   ���M���t(���[�m�[�h)���ǂ�����Ԃ�
        *   @return true(�t)/false(����ȊO)
        **/
        bool isLeaf(void)
        {
            return (left == NULL && right == NULL) ? true : false;
        }
        /**
        *   �[����Ԃ�
        *   @return �[��(�m�[�h��0)
        **/
        int getDepth(void)
        {
            if (isRoot()) return 0;
            else return 1 + parent->getDepth();
        }
    };

private:
    vector<Tree> Forest;    /**< �w�K�t�H���X�g*/
    vector<Data> Samples;   /**< �w�K�T���v��*/

    /**
    *   �����x�N�g�����v�Z����
    *   @param [in] _in �摜�f�[�^
    *   @return �����x�N�g��
    **/
    FeatureVector calcFeature(Mat _in);
    /**
    *   �w�K�T���v����ǂݍ���
    **/
    void LoadSamples(void);
    /**
    *   ��l�����𔭐�������
    *   @return 0.0~1.0�̗���
    **/
    double rand_uniform(void);
    /**
    *   �d���̂Ȃ��������X�g�𐶐�����
    *   @param [in] _size �v�f��
    *   @param [in] _min  ����
    *   @param [in] _max  ���
    **/
    vector<int> make_rand_array_unique(int _size, int _min, int _max);
    /**
    *   �w�K�f�[�^�������_���ɒ��o����
    *   @param [in] _num ���o��
    *   @return ���o�f�[�^
    **/
    vector<Data> getRandomSampleSubset(int _num);
    /**
    *   �p�����[�^�������_���ɐ�������
    *   @return �������ꂽ�p�����[�^
    **/
    SplitParam getRandomParameter(void);
    /**
    *   �f�[�^�̃G���g���s�[���v�Z����
    *   @param [in] _data �v�Z�Ώۃf�[�^
    *   @return �G���g���s�[�l
    **/
    double calcDataEntropy(vector<Data> _data);
    /**
    *   ����֐�
    *   @param [in] _in    ���̓f�[�^
    *   @param [in] _param �p�����[�^
    *   @return true(��)/false(�E)
    **/
    bool splitFunction(Data _in, SplitParam _param);
    /**
    *   �t�̃��x�����݊m�����v�Z
    *   @param [in] _data �f�[�^���X�g
    *   @return ���x���̑��݊m��
    **/
    vector<double> calcProbs(vector<Data> _data);
    /**
    *   �؂��w�K����
    *   @param [out] _tree �w�K�Ώۂ̖�
    *   @param [in]  _data �w�K�Ɏg���T���v���f�[�^
    **/
    void LearnTree(Tree *_tree, vector<Data> _data);
    /**
    *   �؂̎��ʌ��ʂ�Ԃ�
    *   @param [in] _tree �Ώۖ�
    *   @param [in] _fv ���͓����x�N�g��
    *   @param [out] _res ���ʌ���
    **/
    void TestTree(Tree *_tree,FeatureVector _fv, vector<double> &_res);
public:
    RandomForest();     /**< �R���X�g���N�^*/
    ~RandomForest();    /**< �f�X�g���N�^*/
    /**
    *   �X���w�K����
    **/
    void LearnForest();
    /**
    *   ���ʂ���
    *   @param [in] _img ���͉摜
    *   @param [out] _label ���ʌ���
    **/
    void TestForest(Mat _img,int &_label);
};