package ldaCore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

import utils.IntDoubleTuple;
import utils.IntDoubleTupleComperator;

public class OnlineLDA2 {
	
	//
	boolean printLambda = false;
	boolean printGamma  = false;
	boolean printPhi    = false;

	//
	private Map<String, Integer> vocab = new HashMap<String, Integer>();
	private Map<Integer, String> rVocab = new HashMap<Integer, String>();
	
	// Free Parameters
	private int K_;	
	
	// 
	private int D_;	
	private int tmpTotalD_ = 11102;
	private double accTotalD = 0;
	private int accTotalWords = 0;
	private int[] Nds;
	
	//
	double[][][] phi_;
	double[][]   gamma_;
	HashMap<Integer, double[]> lambda_; 
	
	//
	private int[][] ids;	
	private float[][] cts;
	
	// Random Variables
	GammaDistribution gd;
	
	// CONSTANTS
	private double SHAPE = 100d;
	private double SCALE = 1d / SHAPE;
	
	private double DELTA = 1E-5;
	
	private double tau0_ = 1020;
	private double kappa_= 0.7;
	private double rhot;
	private static double alpha_ = 1/2d;
	private double eta_= 1/ 20d;
	
	private int dummySize = 100;

	private double[][] dummyLambdas;
	
	private int batchSize_;
	
	private Set<String> stopWordSet = new HashSet<String>();
	
	//
	private ArrayList<String> newWords;
	
	private String[] Symbols = {"\\", "/", ">" ,"<" ,"-" ,"," ,"." ,"(" ,")" ,":" ,";" ,"'" ,"[" ,"]","!" ,"*" ,"#" ,"+","%" ,"@","&","?","$","0","1","2","3","4","5","6","7","8","9","\t","_","{","}","=","|"};
	
	// Constructor
	public OnlineLDA2(int K, double alpha, double eta, int totalD, double tau0, double kappa, int batchSize, String tmpStopWord){
		// Initialize Free Params
		K_ = K;
		alpha_ = alpha;
		eta_ = eta;
		tmpTotalD_ = totalD;
		tau0_ = tau0;
		kappa_ = kappa;
		batchSize_= batchSize;
		
		// Initialize Internal Params
		setRandomParams();
		setParams();
		setDummyLambda();
		setStopWord(tmpStopWord);
	}

	public OnlineLDA2(int K, double alpha, double eta, int totalD, double tau0, double kappa, int batchSize){

		String tmpStopWord = "";

		K_ = K;
		alpha_ = alpha;
		eta_ = eta;
		tmpTotalD_ = totalD;
		tau0_ = tau0;
		kappa_ = kappa;
		batchSize_= batchSize;
		
		// Initialize Internal Params
		setRandomParams();
		setParams();
		setDummyLambda();
		setStopWord(tmpStopWord);
	}

	private void setStopWord(String stopWord) {
		String[] tmpStopWords = stopWord.split(" "); 
		for(int w=0, SIZE = tmpStopWords.length; w<SIZE; w++){
			stopWordSet.add(tmpStopWords[w]);
		}
	}

	private void setDummyLambda() {
		dummyLambdas = new double[dummySize][];
		for(int b=0; b<dummySize; b++){
			double[] tmpDummyLambda = new double[K_];
			tmpDummyLambda= gd.sample(K_); 
			dummyLambdas[b] = tmpDummyLambda;
		}
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

	private void updateIdCts(int d, Float[] values, String[] labels, int removedLabel) {
		int validWordSize = labels.length - removedLabel;
		Nds[d] = validWordSize;
		int[] tmpIds = new int[validWordSize];
		float[] tmpCts = new float[validWordSize];
		
		int tmpId = 0;
		for(int w=0; w<labels.length; w++){
			if(vocab.containsKey(labels[w])){
				tmpIds[tmpId] = vocab.get(labels[w]);
				tmpCts[tmpId] = values[w];
				tmpId++;
			}
		}
		ids[d] = tmpIds;
		cts[d] = tmpCts;
	}

	private boolean checkStopWord(String label) {
		if(stopWordSet.contains(label)){
			return true;
		}else{
			return false;
		}
	}
	
	private void getIdsCts(String[][] miniBatch) {
		// global
		ids = new int[D_][];
		cts = new float[D_][];
		
		newWords = new ArrayList<String>();
		
		// HashMap of word per Documents
		Map<Integer, Float> labelValueMap = new HashMap<Integer, Float>();
		
		for(int d=0; d<D_; d++){
			// get Nd : number of word per documents
			int Nd = miniBatch[d].length;
			// reset wordCountMap
			labelValueMap = new HashMap<Integer, Float>();
			
			String[] labels = new String[Nd];
			Float[]  values = new Float[Nd];
			
			int removedLabel = 0;

			for(int w=0; w<Nd; w++){
				String[] labelValue = miniBatch[d][w].split(":");
				if(labelValue.length==1){
					continue;
				}
				String label = labelValue[0];
				labels[w] = label;
				Float value  = (float)Double.parseDouble(labelValue[1]);
				values[w] = value;

				label = removeSymbol(label);
				
				if(label.length()==0){
					removedLabel++;
				}

				if(checkStopWord(label)){
					values[w] = 0f;
				}
				
				if(!vocab.containsKey(label)){
					int tmpVSize = vocab.size();
					vocab.put(label, tmpVSize);
					rVocab.put(tmpVSize, label);

					newWords.add(label);
				}
				
				int tmpId = vocab.get(label);
				if(!labelValueMap.containsKey(tmpId)){
					labelValueMap.put(tmpId, 1f);
				}else{
					float tmpValue= labelValueMap.get(tmpId);
					tmpValue += value;
					labelValueMap.put(tmpId, tmpValue);
				}
			}
			updateIdCts(d, values, labels, removedLabel);
		}
	}

	private String removeSymbol(String label) {
		for(String symbol:Symbols){
			label = label.replace(symbol, "");
		}
		
		return label;
	}

	private void getMiniBatchParams(String[][] miniBatch) {
		//
		D_ = miniBatch.length;
		Nds = new int[D_];
		
		for(int d=0; d<D_; d++){
			Nds[d] = miniBatch[d].length;
			// TODO different a little
			accTotalWords += Nds[d];
		}
	}

	private void updateSizeOfParameterPerMiniBatch() {
		// phi_
		phi_ = new double[D_][][];
		gamma_ = new double[D_][];
		// gamma and phi
		for(int d=0; d<D_; d++){
			// Gamma
			double[] gammad = getRandomGammaArray();
			// phi_ not needed to be initialized
			double[][] phid = new double[Nds[d]][K_];

			gamma_[d] = gammad;
			phi_[d]   = phid;
		}
		
		// lambda
		for(String newWord: newWords){
			int tmpId = vocab.get(newWord);
			double[] lambdaNW = getRandomGammaArray(); 
			
			int tmpLambdaIdx = lambda_.size() % dummySize;
			
			for(int k=0; k<K_; k++){
				lambdaNW[k] *= dummyLambdas[tmpLambdaIdx][k];
			}
			lambda_.put(tmpId, lambdaNW);
		}
	}

	private double[] getRandomGammaArray() {
		double[] ret = new double[K_];
		ret = gd.sample(K_);
		return ret;
	}

	private void do_m_step(HashMap<Integer, Float> ntw) {
		// calculate lambdaBar
		HashMap<Integer, double[]> lambdaBar = new HashMap<Integer, double[]>();

		double multiplier = (tmpTotalD_ / D_);
		for(int d=0; d<D_; d++){
			for(int w=0; w<Nds[d]; w++){
				int tmpId = ids[d][w];
				if(!lambdaBar.containsKey(tmpId)){
					double[] tmp = new double[K_];
					for(int k=0; k<K_; k++){
						tmp[k] = eta_ + multiplier * phi_[d][w][k];
					}
					lambdaBar.put(tmpId, tmp);
				}else{
					double[] tmp = lambdaBar.get(tmpId);
					for(int k=0; k<K_; k++){
						tmp[k] += multiplier * phi_[d][w][k];
					}
					
				}
			}
		}
		
		// update
		double oneMinuxRhot = 1d - rhot;
		for(int key:lambda_.keySet()){
			double[] tmp = lambda_.get(key);
			if(!lambdaBar.containsKey(key)){
				for(int k=0; k<K_; k++){
					tmp[k] = oneMinuxRhot * tmp[k] + rhot * eta_;
				}
			}else{
				double[] tmpLambda = lambda_.get(key);
				for(int k=0; k<K_; k++){
					tmp[k] = oneMinuxRhot * tmp[k] + rhot * tmpLambda[k];
				}
			}
			lambda_.put(key, tmp);
		}
	}

	private void do_e_step(int d) {
		double[] lastGamma = copyArray(gamma_[d]);
		double[] nextGamma = copyArray(gamma_[d]);

		do{
			lastGamma = copyArray(nextGamma);
			double sumGammad = getSumDArray(gamma_[d]);
			double sumLambdaK;
			for(int k=0; k<K_; k++){
				sumGammad = getSumDArray(gamma_[d]);
				sumLambdaK= getSumLambdaK(d, k);

				// update gamma
				double tmpGamma_dk = alpha_;
				for(int w=0; w<Nds[d]; w++){
					int tmpId = ids[d][w];
//					tmpGamma_dk += phi_[d][w][k] * ((cts[d][w]));
					tmpGamma_dk += phi_[d][w][k] * ((cts[d][w]));
				}
				gamma_[d][k] = tmpGamma_dk;
				nextGamma[k] = tmpGamma_dk;

				// update phi
				double tmpPhiSum = 0;

				for(int w=0; w<Nds[d]; w++){
					int tmpId = ids[d][w];

					double tmpEqtheta = -1;
					try{
						tmpEqtheta = Gamma.digamma(gamma_[d][k]) - Gamma.digamma(sumGammad);
					}catch(Exception e){
						System.out.println("Error lambda:" + gamma_[d][k]);
						System.out.println("Error lambdaSum:" + sumGammad);
						e.printStackTrace();
					}

					double tmpEqBeta = -1;
					try{
						tmpEqBeta  = Gamma.digamma(lambda_.get(tmpId)[k]) - Gamma.digamma(sumLambdaK);
					}catch(Exception e){
						System.out.println("Error lambda:" + lambda_.get(tmpId)[k]);
						System.out.println("Error lambdaSum:" + sumLambdaK);
						e.printStackTrace();
					}
					phi_[d][w][k] = Math.exp(tmpEqtheta + tmpEqBeta);

					tmpPhiSum += phi_[d][w][k];
				}
				
//				// TODO tmp
				for(int w=0; w<Nds[d]; w++){
					phi_[d][w][k]/= (tmpPhiSum);
				}
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
		
		ret = Math.exp((-1) * (bound / accTotalWords));
		
		return ret;
	}
	
	public int[] getMaxGammaGroup(int D){
		int[] ret = new int[D];
		
		for(int d=0; d<D; d++){
			int tmpK = -1;
			double tmpGammaK = -1;
			for(int k=0; k<K_; k++){
				if(tmpGammaK < gamma_[d][k]){
					tmpK = k;
					tmpGammaK = gamma_[d][k];
				}
			}
			ret[d] = tmpK;
		}
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
					try{
						EqlogTheta_dk = Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum[d]);
						EqlogBeta_kw  = Gamma.digamma(lambda_.get(ids[d][w])[k]) - Gamma.digamma(lambdaSum[k]);
					}catch(Exception e){
						System.out.println("gamma[d][k]:" + gamma_[d][k]);
					}
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
				
				tmpSum3 += (-1) * (Gamma.logGamma(Math.abs(tmpSum3_1 + tmpSum3_2)));
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

	public void trainMiniBatch(String[][] miniBatch, int time) {

		D_ = miniBatch.length;
		
		rhot = Math.pow(tau0_ + time, -kappa_);
	
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
		
		HashMap<Integer, Float> ntw = new HashMap<Integer, Float>(); 
		ntw = getNtw(ntw);
		
//		// check
		updateSizeOfParameterPerMiniBatch();
		

		for(int d=0; d<D_; d++){
			do_e_step(d);
		}
		
		// DO M Step
		do_m_step(ntw);
		
		if(printGamma){
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

	private HashMap<Integer, Float> getNtw(HashMap<Integer, Float> ret) {
		for(int d=0; d<D_; d++){
			for(int w=0; w<Nds[d]; w++){
				if(!ret.containsKey(ids[d][w])){
					ret.put(ids[d][w], cts[d][w]);
				}else{
					float tmp = ret.get(ids[d][w]);
					ret.put(ids[d][w], tmp + cts[d][w]);
				}
			}
		}
		return ret;
	}
}