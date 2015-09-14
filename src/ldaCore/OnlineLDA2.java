package ldaCore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

import utils.IntDoubleTuple;
import utils.IntDoubleTupleComperator;


public class OnlineLDA2 {
	
	//
	boolean printLambda = false;
	boolean printGamma  = false;
	boolean printPhi    = true;

	//
	private Map<String, Integer> vocab = new HashMap<String, Integer>();
	private Map<Integer, String> rVocab = new HashMap<Integer, String>();
	
	// Free Parameters
	private int K_;	
	
	// 
	private int D_;	
	private int tmpTotalD_ = 1024;
	private double accTotalD = 0;
	private int accTotalWords = 0;
	private int[] Nds;
	
	//
	double[][][] phi_;
	double[][]   gamma_;
	HashMap<Integer, double[]> lambda_; 
	
	//
	private int[][] ids;	
	private int[][] cts;
	
	// Random Variables
	GammaDistribution gd;
	
	// CONSTANTS
	private double SHAPE = 100d;
	private double SCALE = 1d / SHAPE;
	
	private double DELTA = 1E-10;
	
	private double tau0_ = 10240;
	private double kappa_= 0.7;
	private double rhot;
	private static double alpha_ = 1/20d;
	private double eta_= 1/ 20d;
	
	//
	private ArrayList<String> newWords;
	
	// Constructor
	public OnlineLDA2(int K, double alpha, double eta, int totalD, double tau0, double kappa){
		// Initialize Free Params
		K_ = K;
		alpha_ = alpha;
		eta_ = eta;
		tmpTotalD_ = totalD;
		tau0_ = tau0;
		kappa_ = kappa;
		
		// Initialize Internal Params
		setRandomParams();
		setParams();
	}
	
	
	private void setParams() {
		lambda_ = new HashMap<Integer, double[]>();
	}


	private void setRandomParams() {
		gd = new GammaDistribution(SHAPE, SCALE);
		gd.reseedRandomGenerator(1001);
		for(int i=0; i<1000; i++){
			double[] tmp = gd.sample(20);
			tmp[0] = tmp[1];
		}
	}


	public void getIdsCts(String[][] miniBatch){
		// global
		ids = new int[D_][];
		cts = new int[D_][];
		
		newWords = new ArrayList<String>();
		
		
		// HashMap of word per Documents
		Map<Integer, Integer> wordCountMap = new HashMap<Integer, Integer>();
		
		for(int d=0; d<D_; d++){
			// get Nd : number of word per documents
			int Nd = miniBatch[d].length;
			// reset wordCountMap
			wordCountMap = new HashMap<Integer, Integer>();

			for(int w=0; w<Nd; w++){
				String tmpWord = miniBatch[d][w];
				if(!vocab.containsKey(tmpWord)){
					int tmpVSize = vocab.size();
					vocab.put(tmpWord, tmpVSize);
					rVocab.put(tmpVSize, tmpWord);

					newWords.add(tmpWord);
				}
				
				int tmpId = vocab.get(tmpWord);
				if(!wordCountMap.containsKey(tmpId)){
					wordCountMap.put(tmpId, 1);
				}else{
					int tmpFrequency = wordCountMap.get(tmpId);
					tmpFrequency++;
					wordCountMap.put(tmpId, tmpFrequency);
				}
			}
			updateIdCts(d, wordCountMap, miniBatch[d]);
		}
	}

	private void updateIdCts(int d, Map<Integer, Integer> wordCountMap, String[] oneBatch) {
		int[] tmpIds = new int[oneBatch.length];
		int[] tmpCts = new int[oneBatch.length];
		
		for(int w=0; w<oneBatch.length; w++){
			tmpIds[w] = vocab.get(oneBatch[w]); 
			tmpCts[w] = wordCountMap.get(tmpIds[w]);
		}

		ids[d] = tmpIds;
		cts[d] = tmpCts;
	}

	public void trainMiniBatch(String[][] miniBatch, int time){

		// TODO 
		if(printLambda){
			System.out.println("Lambda:");
			for(int key: lambda_.keySet()){
				System.out.println(Arrays.toString(lambda_.get(key)));
			}
		}
		
		// get the number of words(Nd) for each documents
		getMiniBatchParams(miniBatch);
		accTotalD += D_;
		// get ids and cts
		getIdsCts(miniBatch);
		
		// check
		updateSizeOfParameterPerMiniBatch(miniBatch);

		rhot = Math.pow(tau0_+time, -kappa_);

		for(int d=0; d<D_; d++){
			// get ids and cts
			do_e_step(d);

			// update Lambda
			do_m_step(d);
		}
		
		// TODO 
//		System.out.println("Gamma:" + Arrays.toString(gamma_[0]));
//		System.out.println("Gamma:" + Arrays.toString(gamma_[1]));
		if(printLambda){
			System.out.println("Gamma:");
			for(int d=0; d<D_; d++){
				for(int w=0; w<gamma_[d].length; w++){
					System.out.print(gamma_[d][w] + ",");
				}
				System.out.println("");
			}
		}
		
		if(printPhi){
			System.out.println("phi");
			for(int d=0; d<D_; d++){
				for(int w=0; w<Nds[d]; w++){
					System.out.println(Arrays.toString(phi_[d][w]));
				}
			}
		}
	}

	private void getMiniBatchParams(String[][] miniBatch) {
		//
		D_ = miniBatch.length;
		Nds = new int[D_];
		
		for(int d=0; d<D_; d++){
			Nds[d] = miniBatch[d].length;
			accTotalWords += Nds[d];
		}
	}

	private void updateSizeOfParameterPerMiniBatch(String[][] miniBatch) {
		// phi_
		phi_ = new double[D_][][];
		gamma_ = new double[D_][];
		// gamma, lambda: None

		// gamma and phi
		for(int d=0; d<D_; d++){
			// Gamma
			double[] gammad = getRandomGammaArray();
			for(int k=0; k<K_; k++) gammad[k] = alpha_ * gammad[d];
			// phi_ not needed to be initialized
			double[][] phid = new double[Nds[d]][K_];

			gamma_[d] = gammad;
			phi_[d]   = phid;
		}
		
		// lambda
		for(String newWord: newWords){
			int tmpId = vocab.get(newWord);
			double[] lambdaNW = getRandomGammaArray(); 
			lambda_.put(tmpId, lambdaNW);
		}
	}

	private double[] getRandomGammaArray() {
		double[] ret = new double[K_];
		ret = gd.sample(K_);
		return ret;
	}

	private void do_m_step(int d) {
		
		HashMap<Integer, double[]> lambdaBar = new HashMap<Integer, double[]>();
		
		double lambdaBar_kw = 0;

		for(int w=0; w<Nds[d]; w++){
			for(int k=0; k<K_; k++){
				int tmpId = ids[d][w];

				lambdaBar_kw = eta_ + (tmpTotalD_ / D_) * cts[d][w] * phi_[d][w][k];
				if(!lambdaBar.containsKey(tmpId)){
					double[] tmpDArray = new double[K_];
					Arrays.fill(tmpDArray, 0d);
					tmpDArray[k] = lambdaBar_kw;
					lambdaBar.put(tmpId, tmpDArray);
				}else{
					double[] tmpDArray= lambdaBar.get(tmpId);
					tmpDArray[k] += lambdaBar_kw;
					lambdaBar.put(tmpId, tmpDArray);
				}
			}
		}
		
		double oneMinusRhot = 1 - rhot;
		for(int lambdaKey:lambda_.keySet()){
			if(!lambdaBar.containsKey(lambdaKey)){
				// TODO 
//				double[] lambdaW = lambda_.get(lambdaKey);
//				for(int k=0; k<K_; k++){
//					lambdaW[k] = oneMinusRhot * lambdaW[k];
//				}
//				lambda_.put(lambdaKey, lambdaW);
			}else{
				double[] lambdaW = lambda_.get(lambdaKey);
				double[] lambdaBarW = lambdaBar.get(lambdaKey);
				for(int k=0; k<K_; k++){
					lambdaW[k] = oneMinusRhot * lambdaW[k] + rhot * lambdaBarW[k];
				}
				lambda_.put(lambdaKey, lambdaW);
			}
		}
	}

	private void do_e_step(int d) {
		double[] lastGamma = copyArray(gamma_[d]);
		double[] nextGamma = copyArray(gamma_[d]);
//		int tmpT = 0;
		do{
			// TODO remove
//			System.out.println("tmpT:" + tmpT++);
			lastGamma = copyArray(nextGamma);
			double sumGammad = getSumDArray(gamma_[d]);
			double sumLambdaK;
			for(int k=0; k<K_; k++){
				sumGammad = getSumDArray(gamma_[d]);
				sumLambdaK= getSumLambdaK(d, k);

				// update phi
				for(int w=0; w<Nds[d]; w++){
					int tmpId = ids[d][w];
					double tmpEqtheta = Gamma.digamma(gamma_[d][k]) - Gamma.digamma(sumGammad);
					double tmpEqBeta  = Gamma.digamma(lambda_.get(tmpId)[k]) - Gamma.digamma(sumLambdaK);
					phi_[d][w][k] = Math.exp(tmpEqtheta + tmpEqBeta);
				}
				
				// update gamma
				double tmpGamma_dk = alpha_;
				for(int w=0; w<Nds[d]; w++){
					tmpGamma_dk += phi_[d][w][k] * cts[d][w];
				}
				gamma_[d][k] = tmpGamma_dk;
				nextGamma[k] = tmpGamma_dk;
			}
		}while(!diffGamma(lastGamma, nextGamma));
	}


	private double getSumLambdaK(Integer d, int k) {
		double ret =0;
		for(int id:lambda_.keySet()){
			ret += lambda_.get(id)[k];
		}
		return ret;
	}


	private double getSumDArray(double[] ds) {
		double ret = 0;
		for(int k=0; k<K_; k++){
			ret += ds[k];
		}
		return ret;
	}


	private boolean diffGamma(double[] lastGamma, double[] nextGamma) {
		double diff = 0;
		for(int k=0; k<K_; k++){
			diff += Math.abs(lastGamma[k] - nextGamma[k]);
		}
		
		if(diff < K_ * DELTA){
			// TODO diff
//			System.out.println("diff:" + diff);
			return true;
		}else{
			return false;
		}
	}

	private double[] copyArray(double[] ds) {
		double[] ret = new double[K_];
		for(int k=0; k<K_; k++){
			ret[k] = ds[k];
		}
		return ret;
	}
	
	public void showTopicWords() {
		System.out.println("SHOW TOPIC WORDS:");
		System.out.println("WORD SIZE:" + lambda_.size());
		for(int k=0; k<K_; k++){
			
			double lambdaSum = 0;
			for(int key:lambda_.keySet()){
				lambdaSum += lambda_.get(key)[k];
			}

			
			System.out.print("Topic:" + k);

			System.out.println("===================================");
			ArrayList<String> sortedWords = getSortedLambda(k);
			System.out.println("k:" + k + " sortedWords.size():" + sortedWords.size());
			for(int tt=0; tt<50; tt++){
				String tmpWord = sortedWords.get(tt);
				int tmpId = vocab.get(tmpWord);
				System.out.println("No." + tt + "\t" +tmpWord + "[" + tmpWord.length() + "]" + ":\t" + lambda_.get(tmpId)[k] / lambdaSum);
			}
			System.out.println("==========================================");
		}
	}
	private ArrayList<String> getSortedLambda(int k) {
		ArrayList<String> ret = new ArrayList<String>();
		ArrayList<IntDoubleTuple> compareList = new ArrayList<IntDoubleTuple>();
		for(int id:lambda_.keySet()){
			double tmpValue = lambda_.get(id)[k];
			compareList.add(new IntDoubleTuple(id, tmpValue));
		}

		Collections.sort(compareList, new IntDoubleTupleComperator());
		
		for(int w=0,W=compareList.size(); w<W; w++){
			int tmpId = compareList.get(w).getId();
			ret.add(rVocab.get(tmpId));
		}
		return ret;
	}	
	
	public double getPerplexity(){
		double ret = 0;
		double bound = calcBoundPerMiniBatch();
		
//		ret = Math.exp((-1) * (bound / accTotalWords));
		ret = bound / accTotalWords;
		
		
		accTotalD = 0;
		accTotalWords = 0;
		
		return ret;
	}
	
	private double calcBoundPerMiniBatch(){
		double ret = 0;
		
		double tmpSum1 = 0;
		double tmpSum2 = 0;
		double tmpSum3 = 0;
		double tmpSum4 = 0;

		double tmpSum3_1 = 0;
		double tmpSum3_2 = 0;
		
		double tmpSum3_2_1 = 0;
		double tmpSum3_2_2 = 0;
		double tmpSum3_2_2_1 = 0;
		double tmpSum3_2_2_2 = 0;
		
		double tmpSum3_2_first = 0;

		double tmpSum3_2_3 = 0;
		
		double tmpSum4_1 = 0;
		double tmpSum4_2 = 0;
		double tmpSum4_3 = 0;
		double tmpSum4_4 = 0;
		
		// Prepare
		double[] gammaSum = new double[D_];
		Arrays.fill(gammaSum, 0d);
		for(int d=0; d<D_; d++){
			for(int k=0; k<K_; k++){
				gammaSum[d] += gamma_[d][k];
			}
		}
		double[] lambdaSum = new double[K_];
		Arrays.fill(lambdaSum, 0d);
		for(int k=0; k<K_; k++){
			for(int key:lambda_.keySet()){
				lambdaSum[k] = lambda_.get(key)[k];
			}
		}

		// Calculate
		for(int d=0; d<D_; d++){

			// FIRST LINE **
			for(int w=0; w<Nds[d]; w++){
				double ndw = cts[d][w];
				double  EqlogTheta_dk = 0;
				double  EqlogBeta_kw = 0;
				tmpSum1 = 0;

				for(int k=0; k<K_; k++){
					EqlogTheta_dk = Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum[d]);
					EqlogBeta_kw  = Gamma.digamma(lambda_.get(ids[d][w])[k]) - Gamma.digamma(lambdaSum[k]);
					tmpSum1 += phi_[d][w][k] * (EqlogTheta_dk + EqlogBeta_kw - Math.log(phi_[d][w][k]));
				}
				ret += ndw * tmpSum1;	// 1-1
			}
			// ** FIRST LINE
			
			// SECOND LINE **
			ret -= (Gamma.logGamma(gammaSum[d]));	// 2-1
			tmpSum2 = 0;
			for(int k=0; k<K_; k++){
				tmpSum2 += (alpha_ - gamma_[d][k]) 
						* (Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum[d]))
						+(Gamma.logGamma(gamma_[d][k]));
				tmpSum2 /= accTotalD;
			}
			ret += tmpSum2;			// 2-2
			// ** SECOND LINE

			// THIRD LINE **
			tmpSum3 = 0;
			for(int k=0; k<K_; k++){
				tmpSum3_1 = 0;
				tmpSum3_2 = 0;
				
				tmpSum3_1 = lambdaSum[k];
				for(int key:lambda_.keySet()){
					tmpSum3_2_1 = (eta_ - lambda_.get(key)[k]);
					
					tmpSum3_2_2_1 = Gamma.digamma(lambda_.get(key)[k]);
					tmpSum3_2_2_2 = Gamma.digamma(lambdaSum[k]);
					tmpSum3_2_2 =  tmpSum3_2_2_1 - tmpSum3_2_2_2;
					
					tmpSum3_2_3 = (Gamma.logGamma(lambda_.get(key)[k]));
					double tmpp = lambda_.get(key)[k];
					
					tmpSum3_2_first = (tmpSum3_2_1 * tmpSum3_2_2);
					tmpSum3_2 += (tmpSum3_2_first + tmpSum3_2_3);
				}
				
				tmpSum3 += (-1) * (Gamma.logGamma(tmpSum3_1 + tmpSum3_2));
				System.out.println(tmpSum3);
			}
			ret += (tmpSum3 / D_);		//3-1
			// ** THIRD LINE
			
			// FOURTH LINE **
			double W = lambda_.size();
			tmpSum4_1 = (Gamma.logGamma(K_ * alpha_));
			tmpSum4_2 = (K_ * (Gamma.logGamma(alpha_)));
			tmpSum4_3 = (Gamma.logGamma(W * eta_));
			tmpSum4_4 = (-1) * W * (Gamma.logGamma(eta_)); 
			tmpSum4 =  tmpSum4_1 - tmpSum4_2 + ((tmpSum4_3 - tmpSum4_4) / accTotalD); 
			ret += tmpSum4;
			// ** FOURTH LINE
		}
		return ret;
	}
}
