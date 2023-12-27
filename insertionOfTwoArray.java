import java.util.ArrayList;
import java.util.Arrays;

public class insertionOfTwoArray {
    public static int[] InsertionO(int[] nums1,int[] nums2){
        int check = -1;
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        ArrayList<Integer> list = new ArrayList<>();
        for(int i=0;i<nums1.length;i++){
            if(check==nums1[i]){
                continue;
            }
            for(int j=0;j<nums2.length;j++){
                if(nums1[i]==nums2[j]){
                    list.add(nums1[i]);
                    check = nums1[i];
                    j=nums2.length-1;
                }
            }
        }

        System.out.println(list);
        int[] array = list.stream().mapToInt(i -> i).toArray();
        return array;
    }
    public static void main(String[] args) {

        int arr[] = {1,2,2,1};
        int brr[] = {1,2};

        InsertionO(arr, brr);
        
    }
}
