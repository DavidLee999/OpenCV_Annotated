/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright( C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
//(including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even ifadvised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace cv
{
namespace ml
{

const double minEigenValue = DBL_EPSILON;

class CV_EXPORTS EMImpl : public EM
{
public:

    int nclusters;
    int covMatType;
    TermCriteria termCrit;

    CV_IMPL_PROPERTY_S(TermCriteria, TermCriteria, termCrit)

    void setClustersNumber(int val)
    {
        nclusters = val;
        CV_Assert(nclusters >= 1);
    }

    int getClustersNumber() const
    {
        return nclusters;
    }

    void setCovarianceMatrixType(int val)
    {
        covMatType = val;
        CV_Assert(covMatType == COV_MAT_SPHERICAL ||
                  covMatType == COV_MAT_DIAGONAL ||
                  covMatType == COV_MAT_GENERIC);
    }

    int getCovarianceMatrixType() const
    {
        return covMatType;
    }

    EMImpl()
    {
        nclusters = DEFAULT_NCLUSTERS;
        covMatType=EM::COV_MAT_DIAGONAL;
        termCrit = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, 1e-6);
    }

    virtual ~EMImpl() {}

    void clear()
    {
        trainSamples.release();
        trainProbs.release();
        trainLogLikelihoods.release();
        trainLabels.release();

        weights.release();
        means.release();
        covs.clear();

        covsEigenValues.clear();
        invCovsEigenValues.clear();
        covsRotateMats.clear();

        logWeightDivDet.release();
    }

    bool train(const Ptr<TrainData>& data, int)
    {
		// 获得样本数据，设置样本类别标签
        Mat samples = data->getTrainSamples(), labels;
		// 从trainEM开始进行EM算法
        return trainEM(samples, labels, noArray(), noArray());
    }

    bool trainEM(InputArray samples, // 训练数据，N*d，N为样本数量，d为样本维度
               OutputArray logLikelihoods, // 每个样本对应的对数似然值，N*1
               OutputArray labels, // 最终的样本类别标签，N*1
               OutputArray probs) // 最终的样本后验概率，N*K，K是高斯分布数量
    {
        Mat samplesMat = samples.getMat();
		// 自动设置EM算法所需数据，第二个参数为样本数据，后四个分别为probs0, means0, covs0, weights0
		// 该4个数据在doTrain()函数中通过k-means算法得到，因此为0
        setTrainData(START_AUTO_STEP, samplesMat, 0, 0, 0, 0);
		// 开始进行训练
        return doTrain(START_AUTO_STEP, logLikelihoods, labels, probs);
    }

	// 从E步开始，进行EM算法
    bool trainE(InputArray samples,
                InputArray _means0, // 高斯分布的均值，K*d
                InputArray _covs0, // 高斯分布的协方差矩阵，d*d
                InputArray _weights0, // 隐变量，各个高斯分布的权值，K*1
                OutputArray logLikelihoods,
                OutputArray labels,
                OutputArray probs)
    {
        Mat samplesMat = samples.getMat();
		// 设置协方差矩阵
        std::vector<Mat> covs0;
        _covs0.getMatVector(covs0);

		// 获得均值和权值
        Mat means0 = _means0.getMat(), weights0 = _weights0.getMat();

		// 设置均值、权值和协方差矩阵
        setTrainData(START_E_STEP, samplesMat, 0, !_means0.empty() ? &means0 : 0,
                     !_covs0.empty() ? &covs0 : 0, !_weights0.empty() ? &weights0 : 0);

		// 从E步开始EM算法
        return doTrain(START_E_STEP, logLikelihoods, labels, probs);
    }

	// 从M步开始EM算法
    bool trainM(InputArray samples,
                InputArray _probs0,
                OutputArray logLikelihoods,
                OutputArray labels,
                OutputArray probs)
    {
        Mat samplesMat = samples.getMat();
		// 得到后验概率
        Mat probs0 = _probs0.getMat();

		// 设置后验概率
		setTrainData(START_M_STEP, samplesMat, !_probs0.empty() ? &probs0 : 0, 0, 0, 0);
		// 从M步开始EM算法
		return doTrain(START_M_STEP, logLikelihoods, labels, probs);
    }

    float predict(InputArray _inputs, OutputArray _outputs, int) const
    {
        bool needprobs = _outputs.needed();
        Mat samples = _inputs.getMat(), probs, probsrow;
        int ptype = CV_64F;
        float firstres = 0.f;
        int i, nsamples = samples.rows;

        if( needprobs )
        {
            if( _outputs.fixedType() )
                ptype = _outputs.type();
            _outputs.create(samples.rows, nclusters, ptype);
            probs = _outputs.getMat();
        }
        else
            nsamples = std::min(nsamples, 1);

        for( i = 0; i < nsamples; i++ )
        {
            if( needprobs )
                probsrow = probs.row(i);
            Vec2d res = computeProbabilities(samples.row(i), needprobs ? &probsrow : 0, ptype);
            if( i == 0 )
                firstres = (float)res[1];
        }
        return firstres;
    }

	// 训练完毕后的预测函数
    Vec2d predict2(InputArray _sample, OutputArray _probs) const
    {
        int ptype = CV_64F;
        Mat sample = _sample.getMat(); // 待预测样本数据
        CV_Assert(isTrained());

		// 样本类型转换
        CV_Assert(!sample.empty());
        if(sample.type() != CV_64FC1)
        {
            Mat tmp;
            sample.convertTo(tmp, CV_64FC1);
            sample = tmp;
        }
        sample = sample.reshape(1, 1); // 将样本数据改为单通道，1*d的向量

        Mat probs;
        if( _probs.needed() ) // 后验概率
        {
            if( _probs.fixedType() )
                ptype = _probs.type();
            _probs.create(1, nclusters, ptype);
            probs = _probs.getMat();
        }

		// 调用后验概率计算函数得到预测结果
        return computeProbabilities(sample, !probs.empty() ? &probs : 0, ptype);
    }

    bool isTrained() const
    {
        return !means.empty();
    }

    bool isClassifier() const
    {
        return true;
    }

    int getVarCount() const
    {
        return means.cols;
    }

    String getDefaultName() const
    {
        return "opencv_ml_em";
    }

    static void checkTrainData(int startStep, const Mat& samples,
                               int nclusters, int covMatType, const Mat* probs, const Mat* means,
                               const std::vector<Mat>* covs, const Mat* weights)
    {
        // Check samples.
        CV_Assert(!samples.empty());
        CV_Assert(samples.channels() == 1);

        int nsamples = samples.rows;
        int dim = samples.cols;

        // Check training params.
        CV_Assert(nclusters > 0);
        CV_Assert(nclusters <= nsamples);
        CV_Assert(startStep == START_AUTO_STEP ||
                  startStep == START_E_STEP ||
                  startStep == START_M_STEP);
        CV_Assert(covMatType == COV_MAT_GENERIC ||
                  covMatType == COV_MAT_DIAGONAL ||
                  covMatType == COV_MAT_SPHERICAL);

        CV_Assert(!probs ||
            (!probs->empty() &&
             probs->rows == nsamples && probs->cols == nclusters &&
             (probs->type() == CV_32FC1 || probs->type() == CV_64FC1)));

        CV_Assert(!weights ||
            (!weights->empty() &&
             (weights->cols == 1 || weights->rows == 1) && static_cast<int>(weights->total()) == nclusters &&
             (weights->type() == CV_32FC1 || weights->type() == CV_64FC1)));

        CV_Assert(!means ||
            (!means->empty() &&
             means->rows == nclusters && means->cols == dim &&
             means->channels() == 1));

        CV_Assert(!covs ||
            (!covs->empty() &&
             static_cast<int>(covs->size()) == nclusters));
        if(covs)
        {
            const Size covSize(dim, dim);
            for(size_t i = 0; i < covs->size(); i++)
            {
                const Mat& m = (*covs)[i];
                CV_Assert(!m.empty() && m.size() == covSize && (m.channels() == 1));
            }
        }

        if(startStep == START_E_STEP)
        {
            CV_Assert(means);
        }
        else if(startStep == START_M_STEP)
        {
            CV_Assert(probs);
        }
    }

    static void preprocessSampleData(const Mat& src, Mat& dst, int dstType, bool isAlwaysClone)
    {
        if(src.type() == dstType && !isAlwaysClone)
            dst = src;
        else
            src.convertTo(dst, dstType);
    }

    static void preprocessProbability(Mat& probs)
    {
        max(probs, 0., probs);

		// 计算均匀概率分布
        const double uniformProbability = (double)(1./probs.cols);
		// 遍历所有样本
        for(int y = 0; y < probs.rows; y++)
        {
            Mat sampleProbs = probs.row(y); // 当前样本的后验概率分布

            double maxVal = 0;
            minMaxLoc(sampleProbs, 0, &maxVal); // 获得最大后验概率分布
			// 如果该样本的最大后验概率小于极小值，则将后验概率改为均匀分布
            if(maxVal < FLT_EPSILON)
                sampleProbs.setTo(uniformProbability);
            else // 否则进行L1归一化处理
                normalize(sampleProbs, sampleProbs, 1, 0, NORM_L1);
        }
    }

    void setTrainData(int startStep, const Mat& samples,
                      const Mat* probs0,
                      const Mat* means0,
                      const std::vector<Mat>* covs0,
                      const Mat* weights0)
    {
        clear(); // 清空全局变量

		// 检查参数是否正确
        checkTrainData(startStep, samples, nclusters, covMatType, probs0, means0, covs0, weights0);

		// 如果是从START_AUTO_STEP或者START_E_STEP开始，且混合高斯分布中的协方差矩阵和权重均为空时，利用k-means算法估计初始化参数
		// 此时，需要提供鲜艳的聚类均值
        bool isKMeansInit = (startStep == START_AUTO_STEP) || (startStep == START_E_STEP && (covs0 == 0 || weights0 == 0));
        // Set checked data
		// 复制samples到trainSamples并设置其类型
        preprocessSampleData(samples, trainSamples, isKMeansInit ? CV_32FC1 : CV_64FC1, false);

        // set probs
		// 如果是从m步开始迭代，则需要提供初始后验概率
        if(probs0 && startStep == START_M_STEP)
        {
			// 设置trainProbs
            preprocessSampleData(*probs0, trainProbs, CV_64FC1, true);
			// 归一化后验概率
            preprocessProbability(trainProbs);
        }

		// 后面则是根据e步来设置各个参数
        // set weights
        if(weights0 && (startStep == START_E_STEP && covs0))
        {
            weights0->convertTo(weights, CV_64FC1);
            weights = weights.reshape(1,1); // 将weights转换为1*K的形式
            preprocessProbability(weights);
        }

        // set means
        if(means0 && (startStep == START_E_STEP/* || startStep == START_AUTO_STEP*/))
            means0->convertTo(means, isKMeansInit ? CV_32FC1 : CV_64FC1);

        // set covs
		// covs为大小为k的数组，元素是d*d的协方差矩阵，需要逐个设置
        if(covs0 && (startStep == START_E_STEP && weights0))
        {
            covs.resize(nclusters);
            for(size_t i = 0; i < covs0->size(); i++)
                (*covs0)[i].convertTo(covs[i], CV_64FC1);
        }
    }

    void decomposeCovs()
    {
        CV_Assert(!covs.empty());
        covsEigenValues.resize(nclusters); // 协方差矩阵的特征值
		// 如果协方差矩阵是全协方差，则进行SVD分解时需要计算完整的Sigma = UWU^T
        if(covMatType == COV_MAT_GENERIC)
            covsRotateMats.resize(nclusters); // U矩阵，特征向量
        invCovsEigenValues.resize(nclusters); // 特征值矩阵的逆矩阵
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            CV_Assert(!covs[clusterIndex].empty());

			// 奇异值分解
            SVD svd(covs[clusterIndex], SVD::MODIFY_A + SVD::FULL_UV);
			
			// 球形协方差矩阵
            if(covMatType == COV_MAT_SPHERICAL)
            {
                double maxSingularVal = svd.w.at<double>(0);
                covsEigenValues[clusterIndex] = Mat(1, 1, CV_64FC1, Scalar(maxSingularVal));
            }
			// 对角协方差矩阵
            else if(covMatType == COV_MAT_DIAGONAL)
            {
                covsEigenValues[clusterIndex] = covs[clusterIndex].diag().clone(); //Preserve the original order of eigen values.
            }
            else //COV_MAT_GENERIC
            {
                covsEigenValues[clusterIndex] = svd.w;
                covsRotateMats[clusterIndex] = svd.u;
            }
			// 避免计算所得的特征值过小，取倒数后得到的值太大
            max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);
            invCovsEigenValues[clusterIndex] = 1./covsEigenValues[clusterIndex];
        }
    }

	// 调用k-means算法对训练样本进行预分类
    void clusterTrainSamples()
    {
        int nsamples = trainSamples.rows; // 样本数量

        // Cluster samples, compute/update means

        // Convert samples and means to 32F, because kmeans requires this type.
        Mat trainSamplesFlt, meansFlt;
        if(trainSamples.type() != CV_32FC1)
            trainSamples.convertTo(trainSamplesFlt, CV_32FC1);
        else
            trainSamplesFlt = trainSamples;
        if(!means.empty())
        {
            if(means.type() != CV_32FC1)
                means.convertTo(meansFlt, CV_32FC1);
            else
                meansFlt = means;
        }

        Mat labels; // 样本的分类标签
		// 调用k-means函数
        kmeans(trainSamplesFlt, nclusters, labels,
               TermCriteria(TermCriteria::COUNT, means.empty() ? 10 : 1, 0.5),
               10, KMEANS_PP_CENTERS, meansFlt);

        // Convert samples and means back to 64F.
        CV_Assert(meansFlt.type() == CV_32FC1);
        if(trainSamples.type() != CV_64FC1)
        {
            Mat trainSamplesBuffer;
            trainSamplesFlt.convertTo(trainSamplesBuffer, CV_64FC1);
            trainSamples = trainSamplesBuffer;
        }
        meansFlt.convertTo(means, CV_64FC1);

        // Compute weights and covs
        weights = Mat(1, nclusters, CV_64FC1, Scalar(0));
        covs.resize(nclusters);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            Mat clusterSamples; // 找出属于某个高斯分布的样本数据
			// 遍历所有样本，将每个样本分给对应的高斯分布
            for(int sampleIndex = 0; sampleIndex < nsamples; sampleIndex++)
            {
                if(labels.at<int>(sampleIndex) == clusterIndex)
                {
                    const Mat sample = trainSamples.row(sampleIndex);
                    clusterSamples.push_back(sample);
                }
            }
            CV_Assert(!clusterSamples.empty());

			// 计算当前高斯分布的均值和方差
            calcCovarMatrix(clusterSamples, covs[clusterIndex], means.row(clusterIndex),
                CV_COVAR_NORMAL + CV_COVAR_ROWS + CV_COVAR_USE_AVG + CV_COVAR_SCALE, CV_64FC1);
			// 计算不同高斯分布的权值
            weights.at<double>(clusterIndex) = static_cast<double>(clusterSamples.rows)/static_cast<double>(nsamples);
        }

		// 对协方差矩阵进行svd分解
        decomposeCovs();
    }

	// 计算log(weight_k) - log(|det(cov_k)|) / 2
    void computeLogWeightDivDet()
    {
        CV_Assert(!covsEigenValues.empty());

        Mat logWeights; // 权值因子的对数值
        cv::max(weights, DBL_MIN, weights); // 确保权值因子不够小
        log(weights, logWeights);

        logWeightDivDet.create(1, nclusters, CV_64FC1);
        // note: logWeightDivDet = log(weight_k) - 0.5 * log(|det(cov_k)|)

		// 遍历所有高斯成分
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            double logDetCov = 0.;
            const int evalCount = static_cast<int>(covsEigenValues[clusterIndex].total()); // 协方值矩阵特征值的总数
            for(int di = 0; di < evalCount; di++)
                logDetCov += std::log(covsEigenValues[clusterIndex].at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0));

            logWeightDivDet.at<double>(clusterIndex) = logWeights.at<double>(clusterIndex) - 0.5 * logDetCov;
        }
    }

    bool doTrain(int startStep, OutputArray logLikelihoods, OutputArray labels, OutputArray probs)
    {
		// EM算法初始化
        int dim = trainSamples.cols; // 训练样本维度
        // Precompute the empty initial train data in the cases of START_E_STEP and START_AUTO_STEP
		// 执行k-means算法获得对应的参数
        if(startStep != START_M_STEP)
        {
            if(covs.empty())
            {
                CV_Assert(weights.empty());
                clusterTrainSamples();
            }
        }

		// 如果未进行上一步操作，就在这步对协方差矩阵进行SVD分解，获得其特征值矩阵
        if(!covs.empty() && covsEigenValues.empty() )
        {
            CV_Assert(invCovsEigenValues.empty());
            decomposeCovs();
        }

		// 如果是从m步快开始，则先由probs^0经由m步得到weights^1, means^1, covs^1
        if(startStep == START_M_STEP)
            mStep();

        double trainLogLikelihood, prevTrainLogLikelihood = 0.; // 对数似然函数的值和前一步所得的对数似然函数的值
		// 停止迭代的条件
        int maxIters = (termCrit.type & TermCriteria::MAX_ITER) ?
            termCrit.maxCount : DEFAULT_MAX_ITERS;
        double epsilon = (termCrit.type & TermCriteria::EPS) ? termCrit.epsilon : 0.;

		// 初始化完毕，开始em算法
        for(int iter = 0; ; iter++)
        {
			// 执行e步
            eStep();
			// 获得当前对数似然函数的值
            trainLogLikelihood = sum(trainLogLikelihoods)[0];

			// 查看是否停止迭代
            if(iter >= maxIters - 1)
                break;

            double trainLogLikelihoodDelta = trainLogLikelihood - prevTrainLogLikelihood;
            if( iter != 0 &&
                (trainLogLikelihoodDelta < -DBL_EPSILON ||
                 trainLogLikelihoodDelta < epsilon * std::fabs(trainLogLikelihood)))
                break;

			// 执行m步
            mStep();

			// 保留当前步的对数似然函数值
            prevTrainLogLikelihood = trainLogLikelihood;
        }

		// 如果对数似然函数值是一个很大的负数，则算法出错，退出函数
        if( trainLogLikelihood <= -DBL_MAX/10000. )
        {
            clear();
            return false;
        }

        // postprocess covs
		// 对协方差矩阵进行后处理，对于球形协方差矩阵，之前只用到了特征值，现在需要计算真的球形协方差矩阵
        covs.resize(nclusters);
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
            if(covMatType == COV_MAT_SPHERICAL)
            {
                covs[clusterIndex].create(dim, dim, CV_64FC1);
				// 单位矩阵
                setIdentity(covs[clusterIndex], Scalar(covsEigenValues[clusterIndex].at<double>(0)));
            }
            else if(covMatType == COV_MAT_DIAGONAL)
            {
				// 对角线矩阵
                covs[clusterIndex] = Mat::diag(covsEigenValues[clusterIndex]);
            }
        }

		// 复制输出
        if(labels.needed())
            trainLabels.copyTo(labels);
        if(probs.needed())
            trainProbs.copyTo(probs);
        if(logLikelihoods.needed())
            trainLogLikelihoods.copyTo(logLikelihoods);

		// 释放内存
        trainSamples.release();
        trainProbs.release();
        trainLabels.release();
        trainLogLikelihoods.release();

        return true;
    }

    Vec2d computeProbabilities(const Mat& sample, Mat* probs, int ptype) const
    {
        // L_ik = log(weight_k) - 0.5 * log(|det(cov_k)|) - 0.5 *(x_i - mean_k)' cov_k^(-1) (x_i - mean_k)]
        // q = arg(max_k(L_ik))
        // probs_ik = exp(L_ik - L_iq) / (1 + sum_j!=q (exp(L_ij - L_iq))
        // see Alex Smola's blog http://blog.smola.org/page/2 for
        // details on the log-sum-exp trick

        int stype = sample.type();
        CV_Assert(!means.empty());
        CV_Assert((stype == CV_32F || stype == CV_64F) && (ptype == CV_32F || ptype == CV_64F));
        CV_Assert(sample.size() == Size(means.cols, 1));

        int dim = sample.cols; // 样本的属性数量

        Mat L(1, nclusters, CV_64FC1), centeredSample(1, dim, CV_64F); // L_ik，1*K
        int i, label = 0;
		// 遍历所有高斯成分
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
			// 计算x - \mu
            const double* mptr = means.ptr<double>(clusterIndex);
            double* dptr = centeredSample.ptr<double>();
            if( stype == CV_32F )
            {
                const float* sptr = sample.ptr<float>();
                for( i = 0; i < dim; i++ )
                    dptr[i] = sptr[i] - mptr[i];
            }
            else
            {
                const double* sptr = sample.ptr<double>();
                for( i = 0; i < dim; i++ )
                    dptr[i] = sptr[i] - mptr[i];
            }

			// 如果是全协方差矩阵，则相减后还需要乘以对应的特征值，即D*U^T；否则可以直接取对应的相减的值，即D
            Mat rotatedCenteredSample = covMatType != COV_MAT_GENERIC ?
                    centeredSample : centeredSample * covsRotateMats[clusterIndex];

            double Lval = 0;
			// 遍历所有属性，计算L的值
            for(int di = 0; di < dim; di++)
            {
                double w = invCovsEigenValues[clusterIndex].at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0);
                double val = rotatedCenteredSample.at<double>(di);
                Lval += w * val * val;
            }
            CV_DbgAssert(!logWeightDivDet.empty());
            L.at<double>(clusterIndex) = logWeightDivDet.at<double>(clusterIndex) - 0.5 * Lval;

			// 获得最大后验概率
            if(L.at<double>(clusterIndex) > L.at<double>(label))
                label = clusterIndex;
        }

		// log-sum-exp技巧
        double maxLVal = L.at<double>(label); // 最大的L值
        double expDiffSum = 0;
		// 遍历所有高斯分布
        for( i = 0; i < L.cols; i++ )
        {
            double v = std::exp(L.at<double>(i) - maxLVal);
            L.at<double>(i) = v;
            expDiffSum += v; // sum_j(exp(L_ij - L_iq)) // 求和，后验概率的归一化参数（分母）
        }

        if(probs)
            L.convertTo(*probs, ptype, 1./expDiffSum);

        Vec2d res;
        res[0] = std::log(expDiffSum)  + maxLVal - 0.5 * dim * CV_LOG2PI; // 对数似然函数，要减去常数d * log(2pi) / 2
        res[1] = label; // 对应的类别标签

        return res;
    }

    void eStep()
    {
        // Compute probs_ik from means_k, covs_k and weights_k.
        trainProbs.create(trainSamples.rows, nclusters, CV_64FC1); // 后验概率，1*K
        trainLabels.create(trainSamples.rows, 1, CV_32SC1); // 类别标签，N*1
        trainLogLikelihoods.create(trainSamples.rows, 1, CV_64FC1); // 对数似然函数，N*1

		// 计算log(weights) - log(|Sigma|) / 2
        computeLogWeightDivDet();

        CV_DbgAssert(trainSamples.type() == CV_64FC1);
        CV_DbgAssert(means.type() == CV_64FC1);

		// 遍历所有样本
        for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
        {
            Mat sampleProbs = trainProbs.row(sampleIndex); // 当前样本的后验概率，1*K
            Vec2d res = computeProbabilities(trainSamples.row(sampleIndex), &sampleProbs, CV_64F); // 计算后验概率
            trainLogLikelihoods.at<double>(sampleIndex) = res[0]; // 当前样本的对数似然函数
            trainLabels.at<int>(sampleIndex) = static_cast<int>(res[1]); // 对应的类别标签
        }
    }

    void mStep()
    {
        // Update means_k, covs_k and weights_k from probs_ik
        int dim = trainSamples.cols; // 样本维度

        // Update weights
        // not normalized first
		// 更新权重，即为对应类别的后验概率之和，这里的权重尚未除以样本数量N，方便后续计算
        reduce(trainProbs, weights, 0, CV_REDUCE_SUM);

        // Update means
        means.create(nclusters, dim, CV_64FC1);
        means = Scalar(0);

		// 先获得一个很大和很小的值
        const double minPosWeight = trainSamples.rows * DBL_EPSILON;
        double minWeight = DBL_MAX;
        int minWeightClusterIndex = -1;
		// 遍历所有高斯分布
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
			// 如果对应的权值因子很小，将之作为分母会得到一个很大的值，先跳过
            if(weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

			// 记录最小的权值及其对应的高斯分布
            if(weights.at<double>(clusterIndex) < minWeight)
            {
                minWeight = weights.at<double>(clusterIndex);
                minWeightClusterIndex = clusterIndex;
            }

			// 获得当前高斯分布的均值
            Mat clusterMean = means.row(clusterIndex);
			// 遍历素有样本，累加计算均值
            for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
                clusterMean += trainProbs.at<double>(sampleIndex, clusterIndex) * trainSamples.row(sampleIndex);
            clusterMean /= weights.at<double>(clusterIndex);
        }

        // Update covsEigenValues and invCovsEigenValues
        covs.resize(nclusters); // 定义协方差矩阵及其特征值
        covsEigenValues.resize(nclusters);
        if(covMatType == COV_MAT_GENERIC)
            covsRotateMats.resize(nclusters); // 特征向量
        invCovsEigenValues.resize(nclusters);
		// 遍历所有高斯分布
        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
			// 如果对应的权值因子很小，将之作为分母会得到一个很大的值，先跳过
            if(weights.at<double>(clusterIndex) <= minPosWeight)
                continue;

			// 创建不同的协方差矩阵
            if(covMatType != COV_MAT_SPHERICAL)
                covsEigenValues[clusterIndex].create(1, dim, CV_64FC1);
            else
                covsEigenValues[clusterIndex].create(1, 1, CV_64FC1);

            if(covMatType == COV_MAT_GENERIC)
                covs[clusterIndex].create(dim, dim, CV_64FC1);

			// 当前高斯分布的协方差矩阵
			// 如果不是全协方差，则只需要由其特征值组成的对角阵，否则需要整个协方差矩阵
            Mat clusterCov = covMatType != COV_MAT_GENERIC ?
                covsEigenValues[clusterIndex] : covs[clusterIndex];

            clusterCov = Scalar(0);

            Mat centeredSample; // 减去对应均值的样本
            for(int sampleIndex = 0; sampleIndex < trainSamples.rows; sampleIndex++)
            {
                centeredSample = trainSamples.row(sampleIndex) - means.row(clusterIndex);

				// 全协方差矩阵，按照公式计算
                if(covMatType == COV_MAT_GENERIC)
                    clusterCov += trainProbs.at<double>(sampleIndex, clusterIndex) * centeredSample.t() * centeredSample;
                else
                {
                    double p = trainProbs.at<double>(sampleIndex, clusterIndex); // p(k|x_i, sigma)，当前样本在当前高斯分布下的后验概率
                    for(int di = 0; di < dim; di++ )
                    {
                        double val = centeredSample.at<double>(di); // 第di个属性
                        clusterCov.at<double>(covMatType != COV_MAT_SPHERICAL ? di : 0) += p*val*val; // 只需要计算对角线
                    }
                }
            }

			// 球形协方差矩阵，特征没有加权（乘以p），因此在最后统一除以d
            if(covMatType == COV_MAT_SPHERICAL)
                clusterCov /= dim;

			// 再除以对应的权值
            clusterCov /= weights.at<double>(clusterIndex);

            // Update covsRotateMats for COV_MAT_GENERIC only
			// 如果是全协方差矩阵，则需要进行svd分解
            if(covMatType == COV_MAT_GENERIC)
            {
                SVD svd(covs[clusterIndex], SVD::MODIFY_A + SVD::FULL_UV);
                covsEigenValues[clusterIndex] = svd.w;
                covsRotateMats[clusterIndex] = svd.u;
            }

			// 防止所得的特征值过小
            max(covsEigenValues[clusterIndex], minEigenValue, covsEigenValues[clusterIndex]);

            // update invCovsEigenValues
            invCovsEigenValues[clusterIndex] = 1./covsEigenValues[clusterIndex];
        }

        for(int clusterIndex = 0; clusterIndex < nclusters; clusterIndex++)
        {
			// 前面对于权值很小的情况直接跳过了，现在倒回来处理这种情况
            if(weights.at<double>(clusterIndex) <= minPosWeight)
            {
                Mat clusterMean = means.row(clusterIndex); // 对应均值
				// 将前面记录的最小权值对应的高斯分布的均值、协方差矩阵等复制给这个分布
                means.row(minWeightClusterIndex).copyTo(clusterMean);
                covs[minWeightClusterIndex].copyTo(covs[clusterIndex]);
                covsEigenValues[minWeightClusterIndex].copyTo(covsEigenValues[clusterIndex]);
                if(covMatType == COV_MAT_GENERIC)
                    covsRotateMats[minWeightClusterIndex].copyTo(covsRotateMats[clusterIndex]);
                invCovsEigenValues[minWeightClusterIndex].copyTo(invCovsEigenValues[clusterIndex]);
            }
        }

        // Normalize weights
        weights /= trainSamples.rows;
    }

    void write_params(FileStorage& fs) const
    {
        fs << "nclusters" << nclusters;
        fs << "cov_mat_type" << (covMatType == COV_MAT_SPHERICAL ? String("spherical") :
                                 covMatType == COV_MAT_DIAGONAL ? String("diagonal") :
                                 covMatType == COV_MAT_GENERIC ? String("generic") :
                                 format("unknown_%d", covMatType));
        writeTermCrit(fs, termCrit);
    }

    void write(FileStorage& fs) const
    {
        writeFormat(fs);
        fs << "training_params" << "{";
        write_params(fs);
        fs << "}";
        fs << "weights" << weights;
        fs << "means" << means;

        size_t i, n = covs.size();

        fs << "covs" << "[";
        for( i = 0; i < n; i++ )
            fs << covs[i];
        fs << "]";
    }

    void read_params(const FileNode& fn)
    {
        nclusters = (int)fn["nclusters"];
        String s = (String)fn["cov_mat_type"];
        covMatType = s == "spherical" ? COV_MAT_SPHERICAL :
                             s == "diagonal" ? COV_MAT_DIAGONAL :
                             s == "generic" ? COV_MAT_GENERIC : -1;
        CV_Assert(covMatType >= 0);
        termCrit = readTermCrit(fn);
    }

    void read(const FileNode& fn)
    {
        clear();
        read_params(fn["training_params"]);

        fn["weights"] >> weights;
        fn["means"] >> means;

        FileNode cfn = fn["covs"];
        FileNodeIterator cfn_it = cfn.begin();
        int i, n = (int)cfn.size();
        covs.resize(n);

        for( i = 0; i < n; i++, ++cfn_it )
            (*cfn_it) >> covs[i];

        decomposeCovs();
        computeLogWeightDivDet();
    }

    Mat getWeights() const { return weights; }
    Mat getMeans() const { return means; }
    void getCovs(std::vector<Mat>& _covs) const
    {
        _covs.resize(covs.size());
        std::copy(covs.begin(), covs.end(), _covs.begin());
    }

    // all inner matrices have type CV_64FC1
    Mat trainSamples;
    Mat trainProbs;
    Mat trainLogLikelihoods;
    Mat trainLabels;

    Mat weights;
    Mat means;
    std::vector<Mat> covs;

    std::vector<Mat> covsEigenValues;
    std::vector<Mat> covsRotateMats;
    std::vector<Mat> invCovsEigenValues;
    Mat logWeightDivDet;
};

Ptr<EM> EM::create()
{
    return makePtr<EMImpl>();
}

Ptr<EM> EM::load(const String& filepath, const String& nodeName)
{
    return Algorithm::load<EM>(filepath, nodeName);
}

}
} // namespace cv

/* End of file. */
