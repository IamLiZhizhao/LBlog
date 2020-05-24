package com.example.demo;

import org.junit.jupiter.api.Test;
import org.mockito.internal.util.StringUtil;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.util.StringUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

@SpringBootTest
class AlgorithmDemoApplicationTests {

	@Test
	void contextLoads() {
		int[] height = {0,1,0,2,1,0,1,3,2,1,2,1};
		int i = trapDP(height);
		System.out.println(i);
	}


    // 42. 接雨水
	public int trap(int[] height) {
		int n = height.length;
		int sum = 0;
		for (int i = 1; i < n-1; i++) {
			int leftMax = 0;
			for (int j = i-1; j >= 0 ; j--) {
				if (height[j] > leftMax) leftMax = height[j];
			}
			int rightMax = 0;
			for (int j = i+1; j < n  ; j++) {
				if (height[j] > rightMax) rightMax = height[j];
			}
			int res = Math.min(leftMax, rightMax);
			sum += height[i] >= res ? 0 : res-height[i];
		}

		return sum;
	}

	// 面试题 17.21. 直方图的水量   dp
	public int trapDP(int[] height) {
		int n = height.length;
		if (n == 0) return 0;
		int sum = 0;
		int[][] dp = new int[n][2];
		dp[0][0] = height[0];
		for (int i = 1; i < n; i++) {
			dp[i][0] = Math.max(dp[i-1][0],height[i]);
		}
		dp[n-1][1] = height[n-1];
		for (int j = n-2; j >= 0; j--) {
			dp[j][1] = Math.max(dp[j+1][1],height[j]);
		}
		for (int i = 1; i < n-1; i++) {
			int res = Math.min(dp[i][0], dp[i][1]);
			sum += height[i] >= res ? 0 : res-height[i];
		}
		return sum;
	}

	@Test
	void context151() {
		String s = "a good   example";
		System.out.println(reverseWords(s));
	}

	//151. 翻转字符串里的单词
	//无空格字符构成一个单词。
	//输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
	//如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
	public String reverseWords(String s) {
		String[] split = s.trim().split(" ");
		StringBuilder sb = new StringBuilder();
		for (int i = split.length-1; i >= 0 ; i--) {
			if (split[i].trim().isEmpty() || split[i].equals(" ")) continue;
			sb.append(split[i]);
			if (i!=0) sb.append(" ");
		}
		return sb.toString();
	}

//memo = dict()
//    def dp(K, N) -> int:
//        # base case
//        if K == 1: return N
//        if N == 0: return 0
//        # 避免重复计算
//        if (K, N) in memo:
//            return memo[(K, N)]
//
//        res = float('INF')
//        # 穷举所有可能的选择
//        for i in range(1, N + 1):
//            res = min(res,
//                      max(
//                            dp(K, N - i),
//                            dp(K - 1, i - 1)
//                         ) + 1
//                  )
//        # 记入备忘录
//        memo[(K, N)] = res
//        return res
//
//    return dp(K, N)

	private int memo[][];

	public int superEggDrop(int K, int N) {
		memo = new int[K+1][N+1];
		for (int i = 1; i < K+1; i++) {
			for (int j = 1; j < N+1; j++) {
				memo[i][j] = -1;
			}
		}
		return dp(K,N);
	}

	private int dp(int K, int N){
		if (K==1) return N;
		if (N==0) return 0;
		if (memo[K][N]!=-1) return memo[K][N];
		int res =0;
		for (int i = 1; i < N+1; i++) {
			res = Math.min(res,Math.max(dp(K,N-i),dp(K-1,i-1))+1);
		}
		memo[K][N] = res;
		return res;
	}


	// 9. 回文数
	public boolean isPalindrome(int x) {
		String value = String.valueOf(x);
		if (value == null || "".equals(value) || value.startsWith("-")) return false;
		StringBuilder sb = new StringBuilder();
		char[] chars = value.toCharArray();
		for (int i = chars.length-1; i >=0 ; i--) {
			sb.append(chars[i]);
		}
		return value.equals(sb.toString());
	}

	// 9. 回文数(进阶：不将整数转为字符串)
	public boolean isPalindrome1(int x) {
		if(x<0 || (x%10==0 && x!=0)) return false;

		int res = 0;
		while (x > res){
			res = res * 10 + x%10;
			x /= 10;
		}

		return x==res || x==res/10;

	}

	//35. 搜索插入位置
	// 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。
	// 如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
	public int searchInsert(int[] nums, int target) {
		int left=0,right = nums.length-1;
		while (left<=right){
			int mid = (left + right)/2;
			if (nums[mid] == target) return mid;
			if (nums[mid] < target){
				left = mid + 1;
			} else {
				right = mid -1;
			}
		}
		return left;
	}

	//13. 罗马数字转整数
	// 字符          数值
	//I             1
	//V             5
	//X             10
	//L             50
	//C             100
	//D             500
	//M             1000
	public int romanToInt(String s) {
		char[] chars = s.toCharArray();
		int left = 0;
		int sum = 0;
		for (int i = 0; i < chars.length; i++) {
			int num = sToRoman(chars[i]);
			if (num > left) sum = sum - left*2 + num;
			else sum += num;
			left = num;
		}

		return sum;
	}

	private int sToRoman(char c){
		int i = 0;
		switch (c){
			case 'I': i = 1; break;
			case 'V': i = 5; break;
			case 'X': i = 10; break;
			case 'L': i = 50; break;
			case 'C': i = 100; break;
			case 'D': i = 500; break;
			case 'M': i = 1000; break;
		}
		return i;
	}

	//给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
	//示例:
	//输入: [-2,1,-3,4,-1,2,1,-5,4],
	//输出: 6
	//解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
	//进阶:
	//如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。

	//1.普通做法：
	public int maxSubArray(int[] nums) {

		int len = nums.length;
		if (len == 0) return 0;
		if (len == 1) return nums[0];

		int maxSum = nums[0];

		for (int i = 0; i < len; i++) {
			int temp = nums[i];
			if (temp > maxSum) maxSum = temp;

			int sum = temp;
			for (int j = i+1; j < len; j++) {
				sum += nums[j];
				if (sum > maxSum) maxSum = sum;
			}
			if (sum!=temp) {
				if (sum > maxSum) maxSum = sum;
			}
		}

		return maxSum;
	}

	//2.进阶做法：
	public int maxSubArray2(int[] nums) {

		int pre = 0, maxAns = nums[0];
		for (int i = 0; i < nums.length; i++) {
			pre = Math.max(pre + nums[i], nums[i]);
			maxAns = Math.max(maxAns, pre);
		}
		return maxAns;
	}

	@Test
	void conTest() {
		int[] nums = new int[]{-1,-2};
		int maxNum = maxSubArray2(nums);

		System.out.println(maxNum);
	}


	//给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
	//如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。
	//注意：你不能在买入股票前卖出股票。
	//示例 1:
	//输入: [7,1,5,3,6,4]
	//输出: 5
	//解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
	//     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。

	//示例 2:
	//输入: [7,6,4,3,1]
	//输出: 0
	//解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
	public int maxProfit(int[] prices) {
		int len = prices.length;
		int maxPro = 0;
		if (len == 0 || len == 1) return maxPro;
		int idx = 0;
		int min = prices[0];
		while(prices[idx] <= min) {
			min = prices[idx];
			idx++;
			if(idx == len) break;
		}

		// dp[] 存的是i 到 最后一个数的最大值
		int dp[] = new int[len];
		int t = len-1;
		while(t >= idx){
			if (t == len-1) {
				dp[t] = prices[t];
			} else {
				dp[t] = Math.max(dp[t+1], prices[t]);
			}
			--t;
		}

		for(int i = idx ;i < len; i++) {
			maxPro = Math.max(maxPro, dp[i] - prices[i-1]);
		}
		return maxPro;
	}

	//123. 买卖股票的最佳时机 III
	//给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。
	//设计一个算法来计算你所能获取的最大利润。你最多可以完成 "两笔" 交易。
	//注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
	//示例 1:
	//输入: [3,3,5,0,0,3,1,4]
	//输出: 6
	//解释: 在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
	//     随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
	public int maxProfit1(int[] prices) {
		int len = prices.length;
		int maxPro = 0;
		if (len == 0 || len == 1) return maxPro;
		int idx = 0;
		int min = prices[0];
		while(prices[idx] <= min) {
			min = prices[idx];
			idx++;
			if(idx == len) break;
		}

		PriorityQueue<Integer> minQueue = new PriorityQueue<>(2);
		int thisCha = 0;
		while(idx < len){
			int cha = prices[idx] - prices[idx-1];
			boolean flag = false;
			if(cha >= 0) {
				thisCha += cha;
			} else{
				flag = true;
			}
			idx++;
			if (flag || idx == len){
				if (minQueue.size() < 2 || thisCha > minQueue.peek()) {
					minQueue.offer(thisCha);
				}
				if (minQueue.size() > 2) {
					minQueue.poll();
				}
				thisCha = 0;
			}
		}
		for (Integer integer : minQueue) {
			maxPro += integer;
		}
		return maxPro;
	}

	//[2,1,2,1,0,1,2]
	@Test
	void conTesdft() {
		int[] nums = new int[]{1,2,4,2,5,7,2,4,9,0};
		int maxNum = maxProfit1(nums);
		System.out.println(maxNum);
	}
}
