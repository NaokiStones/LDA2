package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.math3.special.Gamma;

import ldaCore.OnlineLDA2;

public class Execute {
	
	// Container
	static ArrayList<String> fileNames = new ArrayList<String>();
	static ArrayList<String[][]> stringBatchList = new ArrayList<String[][]>();

	// Free Variables
	private static String baseURI = "/Users/ishikawanaoki/Documents/datasetML/wiki1000_backUp/";
//	private static String baseURI = "/Users/ishikawanaoki/Documents/workspace/LDA/myCorpus/";
	private static int batchSize_ = 2;
	private static int K = 5;
	private static double alpha = 1. / K;
	private static double eta   = 1. / K;
	private static int totalD   = 100;
	private static double tau0  = 4;
	private static double kappa = 0.9;
	
	// Control
	private static int PPLNUM = 2000;

	public static void main(String[] args) {
		
		// IMPORT FILE
		getFiles();

		// MAKE BATCH
		try {
			makeStringBatch();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// generate LDA
		OnlineLDA2 onlineLDA2 = new OnlineLDA2(K, alpha, eta, totalD, tau0, kappa);
		// Train *************************************************
		int time = 0;
		System.out.println("Time MAX:" + stringBatchList.size() * batchSize_);
		
		// FOR PERPLEXITY LOOP
		for(int ppl=0; ppl<PPLNUM; ppl++){
			int tmp = 0;
			for(String[][] stringBATCH:stringBatchList){
				//
//				for(int d=0; d<stringBATCH.length; d++){
//					for(int w=0; w<stringBATCH[d].length; w++){
//						if(stringBATCH[d][w].length()==0){
//							System.out.println("====X X X X X X X X X X X X====");
//						}
//					}
//				}
				//
				tmp++;
				if(tmp >=30)
					break;
				onlineLDA2.trainMiniBatch(stringBATCH, time);
				time += batchSize_;
			}
//			onlineLDA2.showTopicWords();
			System.out.print(/*"perplexity:" +*/ onlineLDA2.getPerplexity() + ",");
//			System.out.print(onlineLDA_Batch.getBound() + ",");
		}	
	}

	private static void makeStringBatch() throws IOException {
		for(int batchIdx=0, BATCHSIZE=fileNames.size(); batchIdx < BATCHSIZE; batchIdx+=batchSize_){
			int tmpBatchSize = -1;
			if(batchIdx + batchSize_ >= BATCHSIZE){
				tmpBatchSize = BATCHSIZE - batchIdx;
			}else{
				tmpBatchSize = batchSize_;
			}

			String[][] tmpStringBatch= new String[tmpBatchSize][];

			for(int tmpLocalBatchIdx=0; tmpLocalBatchIdx<tmpBatchSize; tmpLocalBatchIdx++){
				int tmpBatchIdx = batchIdx + tmpLocalBatchIdx;
				tmpStringBatch[tmpLocalBatchIdx] = makeBatch(tmpBatchIdx);
			}
			stringBatchList.add(tmpStringBatch);
		}
	}
	
	private static String[] makeBatch(int tmpBatchIdx) throws IOException {
		String[] ret = null;
		String targetFileURI = fileNames.get(tmpBatchIdx);	
		
		BufferedReader br;

		try{
			br = new BufferedReader(new FileReader(targetFileURI));
		}catch(FileNotFoundException e){
			System.out.println("targetFileURI:" + targetFileURI);
			throw new FileNotFoundException();
		}

		try{
			while(true){
				String line = br.readLine();
				if(line==null)
					break;
				ret = processLine(line);
			}
		}finally{
			try {
				br.close();
			} catch (IOException e) {
				System.out.println("br close miss");
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e1) {
					e1.printStackTrace();
				}
			}
		}

		return ret;
	}

	private static String[] processLine(String line) {
		
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
		
		
		for(int i=0; i<ret.length; i++){
//			ret[i] = ret[i].replace(" ", "");
//			if(ret[i].length() <= 4){
//				ret[i] = "b";
//			}
		}
		
		ArrayList<String> tmpArrayList = new ArrayList<String>();
		for(int i=0; i<ret.length; i++){
			if(ret[i].length() != 0){
				tmpArrayList.add(ret[i]);
			}
		}
		
		String[] ret2 = new String[tmpArrayList.size()];
		for(int i=0, Size=tmpArrayList.size(); i<Size; i++){
			ret2[i] = tmpArrayList.get(i);
		}
		
		// TODO remove
//		System.out.println("ret2.length:" + ret2.length);

		return ret2;
	}	

	private static void getFiles() {
		
		File tmpFile = new File(baseURI);
		String[] names= tmpFile.list();

		for(int i=0; i<names.length; i++){
			fileNames.add(baseURI + names[i]);	
		}
	}
}
