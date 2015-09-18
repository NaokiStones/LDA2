package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import ldaCore.OnlineLDA2;



public class ExecuteNews20{
	// Container
	static ArrayList<String> fileNames = new ArrayList<String>();
	static ArrayList<String> topicGroup= new ArrayList<String>();
	static ArrayList<Integer> gammaTopics= new ArrayList<Integer>();
	static ArrayList<String[][]> stringBatchList = new ArrayList<String[][]>();
	static HashMap<String, ArrayList<Integer>> topicGammaTopicMap = new HashMap<String, ArrayList<Integer>>();

	// Constant Parameters
	static int batchSize_ = 10;
	static String baseURI;
	static String vocabURI;
	static String tfIdfURI;
	static String countURI;
	
	// limit Read File Per Directory
//	static int limit = 10;

	// LDA Parameters
	static int K = 4;
	static double alpha = 1./(K);
	static double eta = 1./ (K);
	static double tau0 = 80;	// 1024
	static double kappa = 0.8;	// 0.7
	static int IterNum = 100;
	static int PPLNUM = 3;
	static int totalD= (int)11000;

	// Control
	static int limit = 10000;
	static int trainLine=10000;
	
	static OnlineLDA2 onlineLDA2;

	static String stopWord = "a b c d e f g h i j k l m n o p q r s t u v w x y z the of in and have to it was or were this that with is some on for so how you if would com be your my one not never then take for an can no but aaa when as out just from does they back up she those who another her do by must what there at very are am much way all any other me he something someone doesn his also its has into us him than about their may too will had been we them why did being over without these could out which only should even well more where after while anyone our now such under two ten else always going either each however non let done ever between anything before every same since because quite sure here nothing new don off still down yes around few many own go get know think like make say see look use said";
	static String[] stopWords;
	
	
	static boolean tdIdf = true;
	static boolean count = false;
	
	public static void main(String[] args){
		long start = System.nanoTime();
//		baseURI = "/Users/ishikawanaoki/Documents/workspace/LDA/targetData/";
		tfIdfURI= "/Users/ishikawanaoki/Documents/workspace/LDA/vocab/news20vocab.csv";
		countURI= "/Users/ishikawanaoki/Documents/workspace/LDA/vocab/news20wordsCount.csv";
		
	
		stopWords = stopWord.split(" ");	
		onlineLDA2 = new OnlineLDA2(K, alpha, eta, totalD, tau0, kappa, batchSize_);
		
//		getFiles();	 

		try {
//			executeTraining();
			executeTrainingTfIdf();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		long end = System.nanoTime();
		
//		printConfusionMatrix();
		onlineLDA2.showTopicWords();
		
		System.out.println("Experiment Time:" + (end - start));
	}

	private static void executeTrainingTfIdf() throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(countURI));
		// TODO remove?
		BufferedReader br2 = new BufferedReader(new FileReader(tfIdfURI));
		
		int docid = 1;
		ArrayList<ArrayList<String>> tmpBatchWordArrayList = new ArrayList<ArrayList<String>>();
		ArrayList<ArrayList<Float>> tmpBatchIfIdfArrayList = new ArrayList<ArrayList<Float>>();
		ArrayList<String> wordListPerDoc = new ArrayList<String>();
		ArrayList<Float> tfIdfListPerDoc = new ArrayList<Float>();
		
		//
		int tmpDocId = -1;
		String tmpWord= "";
		float tmpIfIdf = -1.0f;

		String tmpLine;
		
		HashMap<String, Float> tdIdfMap = new HashMap<String, Float>();
//		while(true){
//			tmpLine2 = br2.readLine();
//		}


		HashSet<String> stopWordSet = new HashSet<String>();
		for(String tmpStopWord: stopWords){
			stopWordSet.add(tmpStopWord);
		}

		while(true){
			// FetchLine
			tmpLine = null;
			if(count){
				tmpLine= br.readLine();
			}
			if(tdIdf){
				tmpLine = br2.readLine();
			}


			if(tmpLine == null){
				if(tmpBatchIfIdfArrayList.size() != 0){
					int tmpBatchSize = tmpBatchIfIdfArrayList.size();
					String[][] stringBatch = new String[tmpBatchSize][];
					Float[][] tfIdfBatch = new Float[tmpBatchSize][];
					for(int b=0; b<tmpBatchSize; b++){
						int tmpElementSize = tmpBatchIfIdfArrayList.get(b).size();
						String[] stringArray = new String[tmpElementSize];
//						for(int i=0; i<stringArray.length; i++){
//							stringArray[i] = stringArray[i].replace("\"", "");
//						}
						Float[] tfIdfArray   = new Float[tmpElementSize];
						
						for(int w=0; w<tmpElementSize; w++){
							stringArray[w] = tmpBatchWordArrayList.get(b).get(w);
							if(stopWordSet.contains(stringArray[w])){
								tfIdfArray[w] = 0f;
							}else{
								tfIdfArray[w]  = tmpBatchIfIdfArrayList.get(b).get(w);
							}
						}
						stringBatch[b] = stringArray;
						tfIdfBatch[b]  = tfIdfArray;
					}
					onlineLDA2.trainMiniBatch(stringBatch, tfIdfBatch, docid);
				}
				break;
			}
			
			// getParams
			String[] tmpParams = tmpLine.split(",");
			tmpDocId = Integer.parseInt(tmpParams[0]);
			tmpWord  = tmpParams[1];
			tmpIfIdf = Float.parseFloat(tmpParams[2]);
			
			if(tmpDocId == docid){
				wordListPerDoc.add(tmpWord);
				tfIdfListPerDoc.add(tmpIfIdf);
			}else{
				docid++;
				
				tmpBatchWordArrayList.add(wordListPerDoc);
				tmpBatchIfIdfArrayList.add(tfIdfListPerDoc);
				
				wordListPerDoc = new ArrayList<String>();
				tfIdfListPerDoc = new ArrayList<Float>();
				
				wordListPerDoc.add(tmpWord);
				tfIdfListPerDoc.add(tmpIfIdf);
				
				if(tmpBatchIfIdfArrayList.size() == batchSize_){
					String[][] stringBatch = new String[batchSize_][];
					Float[][] tfIdfBatch = new Float[batchSize_][];
					for(int d=0; d<batchSize_; d++){
						int tmpElementSize = tmpBatchIfIdfArrayList.get(d).size();
						String[] stringArray = new String[tmpElementSize];
						Float[] ifIdfArray   = new Float[tmpElementSize];
						for(int w=0; w<tmpElementSize; w++){
							stringArray[w] = tmpBatchWordArrayList.get(d).get(w);
							if(stopWordSet.contains(stringArray[w])){
								ifIdfArray[w] = 0f;
							}else{
								ifIdfArray[w]  = tmpBatchIfIdfArrayList.get(d).get(w);
							}
						}
						
						stringBatch[d] = stringArray;
						tfIdfBatch[d]  = ifIdfArray;
					}
					tmpBatchIfIdfArrayList.clear();
					tmpBatchWordArrayList.clear();
					
					onlineLDA2.trainMiniBatch(stringBatch, tfIdfBatch, docid);
					
//					System.out.print((int)onlineLDA2.getPerplexity() + ", ");
					System.out.println(docid);
				}
			}
		}
		System.out.println("");
	}

	private static void printConfusionMatrix() {

		int[][] confusionMatrix = new int[topicGammaTopicMap.size()][K];
		
		int[] rowSum = new int[topicGammaTopicMap.size()];
		Arrays.fill(rowSum, 0);
		
		int tmpp = 0;
		
		for(String string:topicGammaTopicMap.keySet()){
			ArrayList<Integer> tmpArrayList = topicGammaTopicMap.get(string);
			int[] tmpTopicDist = new int[K];
			Arrays.fill(tmpTopicDist, 0);
			for(int tmpTopic:tmpArrayList){
				tmpTopicDist[tmpTopic]++;
			}
			
			for(int k=0; k<K; k++){
				confusionMatrix[tmpp][k] = tmpTopicDist[k];
				rowSum[tmpp] += tmpTopicDist[k];
			}
			System.out.println("");
			tmpp++;
		}
		
		
		for(int i=0; i<confusionMatrix.length; i++){
			int maxCol = -1;
			double tmpVal = -1;
			for(int j=i; j<K; j++){
				if(tmpVal < confusionMatrix[i][j]){
					maxCol = j;
					tmpVal = confusionMatrix[i][j];
				}
			}
			
			if(i!=maxCol){
				confusionMatrix = swapCol(i, maxCol, confusionMatrix);
			}
		}


		tmpp = 0;
		System.out.println("");
		for(String string:topicGammaTopicMap.keySet()){

			if(string.length() < 25){
				int diff = 25 - string.length();
				for(int ttt=0; ttt<diff; ttt++){
					string = string + " ";
				}
			}
			System.out.print(string + ":\t");
			
			for(int k=0; k<K; k++){
				if(k==0){
					if(rowSum[tmpp] == 0){
						System.out.print(0);	
					}else{
					System.out.print((int)(confusionMatrix[tmpp][0] * 100. / (rowSum[tmpp] * 1.)));
					}
				}else{
					if(rowSum[tmpp] == 0){
						System.out.print("\t" + 0);	
					}else{
						System.out.print("\t" + (int)(confusionMatrix[tmpp][k] * 100. / (rowSum[tmpp] * 1d)));
					}
				}
			}
			System.out.println("");
			tmpp++;
		}
		
	}

	private static int[][] swapCol(int i, int maxCol, int[][] confusionMatrix) {
		int[] iCol = new int[confusionMatrix.length];
		int[] jCol= new int[confusionMatrix.length];
		
		for(int t=0; t<confusionMatrix.length; t++){
			iCol[t] = confusionMatrix[t][i];
			jCol[t] = confusionMatrix[t][maxCol];
		}

		for(int t=0; t<confusionMatrix.length; t++){
			confusionMatrix[t][i]      = jCol[t];
			confusionMatrix[t][maxCol] = iCol[t];
		}
		
		return confusionMatrix;
	}

	private static void executeTraining() throws IOException {
		BufferedReader br;
		String[][] tmpMiniBatch;	
		ArrayList<String[]> tmpBatchList= new ArrayList<String[]>();

		for(int tt=0, SIZE=fileNames.size(); tt< SIZE; tt++){
			
			String tmpURI = fileNames.get(tt);

			br = new BufferedReader(new FileReader(tmpURI));
			int tmp = 0;

			while(true){
				// Line Limitation
				if(tmp > trainLine){
					break;
				}

				if(tmp%1000 ==0){
					//				System.out.println((tmp*1. / trainLine) * 100 + "%");
				}

				// Get Line 
				String line = br.readLine();
				if(line==null)
					break;
				String[] words = processLine(line);
				tmpBatchList.add(words);

				// train
				if(tmpBatchList.size() == batchSize_){
					tmpMiniBatch = new String[batchSize_][];
					for(int d=0; d<batchSize_; d++){
						tmpMiniBatch[d] = tmpBatchList.get(d);
					}

					onlineLDA2.trainMiniBatch(tmpMiniBatch, tmp);
					int[] tmpGroup = onlineLDA2.getMaxGammaGroup(tmpMiniBatch.length);
					for(int t=0; t<tmpGroup.length; t++){
						String originTopic = topicGroup.get(tt);
						topicGammaTopicMap.get(originTopic).add(tmpGroup[t]);
						// TODO remove
//						gammaTopics.add(tmpGroup[t]);
					}
					tmpBatchList.clear();
				}

//				if(tmp%100 == 0){
//					System.out.printf("%.2f,",onlineLDA2.getPerplexity());
//				}

				// INCREMENT
				tmp++;
			}
		}
	}

	private static void getFiles() {
		File parentDir = new File(baseURI);
		String[] childDirNames = parentDir.list();
		
		//
		for(int t=0; t<childDirNames.length; t++){
			topicGammaTopicMap.put(childDirNames[t], new ArrayList<Integer>());
		}

			
		for(String childDirName:childDirNames){
			if(childDirName.startsWith(".")){
				continue;
			}
			String childDirURI = baseURI + "/" + childDirName;

			File childDir = new File(childDirURI);
			String[] childFileNames = childDir.list();

			int cnt = 0;
			for(String childFileName:childFileNames){
				if(limit != -1){
					if(cnt >= limit)
						continue;
				}
				if(childFileName.startsWith(".")){
					continue;
				}
				String childFileURI = childDirURI + "/" + childFileName;
				File childFile = new File(childFileURI);
				if(childFile.exists() && childFile!=null){
					fileNames.add(childFileURI);
					topicGroup.add(childDirName);
				}else{
					System.out.println("File:" + childFileURI + " does not exist!");
					try {
						Thread.sleep(100000);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}
	}

	private static String[] processLine(String line) {
		
		String stopWord = "a b c d e f g h i j k l m n o p q r s t u v w x y z the of in and have to it was or were this that with is some on for so how you if would com be your my one not never then take for an can no but aaa when as out just from does they back up she those who another her do by must what there at very are am much way all any other me he something someone doesn his also its has into us him than about their may too will had been we them why did being over without these could out which only should even well more where after while anyone our now such under two ten else always going either each however non let done ever between anything before every same since because quite sure here nothing new don off still down yes around few many own go get know think like make say see look use said";
		String[] stopWords = stopWord.split(" ");
		HashSet<String> stopWordSet = new HashSet<String>();
		for(String tmpStopWord: stopWords){
			stopWordSet.add(tmpStopWord);
		}

		line = line.toLowerCase();
		
		line = line.replace("\"", " ");
		line = line.replace("\\", " ");
		line = line.replace("/", " ");
		line = line.replace(">", " ");
		line = line.replace("<", " ");
		line = line.replace("-", " ");
		line = line.replace(",", " ");
		line = line.replace(".", " ");
		line = line.replace("(", " ");
		line = line.replace(")", " ");
		line = line.replace(":", " ");
		line = line.replace(";", " ");
		line = line.replace("'", " ");
		line = line.replace("[", " ");
		line = line.replace("]", " ");
		line = line.replace("!", " ");
		line = line.replace("*", " ");
		line = line.replace("#", " ");
		line = line.replace("+", " ");
		line = line.replace("%", " ");
		line = line.replace("@", " ");
		line = line.replace("&", " ");
		line = line.replace("?", " ");
		line = line.replace("$", " ");
		line = line.replace("0", " ");
		line = line.replace("1", " ");
		line = line.replace("2", " ");
		line = line.replace("3", " ");
		line = line.replace("4", " ");
		line = line.replace("5", " ");
		line = line.replace("6", " ");
		line = line.replace("7", " ");
		line = line.replace("8", " ");
		line = line.replace("9", " ");
		line = line.replace("\t", " ");
		line = line.replace("_", " ");
		line = line.replace("{", " ");
		line = line.replace("}", " ");
		line = line.replace("=", " ");
		line = line.replace("|", " ");
		
		
		String[] ret;
		
		
		ret = line.split(" ");
		
		if(line.equals("")){
			ret = new String[1];
			ret[0] = "a";
			return ret;
		}
		
		ArrayList<String> tmpArrayList = new ArrayList<String>();
		for(int i=0; i<ret.length; i++){
			if(ret[i].length() >= 0 && !stopWordSet.contains(ret[i])){
				tmpArrayList.add(ret[i]);
			}
		}
		
		String[] ret2 = new String[tmpArrayList.size()];
		for(int i=0, Size=tmpArrayList.size(); i<Size; i++){
			ret2[i] = tmpArrayList.get(i);
		}

		return ret2;
	}
}
