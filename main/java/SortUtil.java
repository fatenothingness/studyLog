public  class SortUtil {

    /**
     * 快速排序
     */
    public void fastSort(int[] arr){
        process(arr,0,arr.length-1);
    }
    private void process(int[] arr,int L,int R){
        if(L>=R){
            return;
        }
        int [] a = partition(arr,L,R);
        process(arr,L,a[0]-1);
        process(arr,a[1]+1,R);

    }
    private  int[] partition(int[] arr,int L,int R){
        int less = L-1;
        int most = R;
        int num = arr[R];
        int i = L;
        while(i<most){
            if(arr[i]==num){
                i++;
            }else if(arr[i]<num){
                swap(arr,++less,i);
                i++;
            }else {
                swap(arr,--most,i);
            }
        }
        swap(arr,R,most);
        return new int[]{less+1,most};
    }

    public static void swap(int[] arr,int a,int b){
        int tmp = arr[a];
        arr[a] = arr[b];
        arr[b] = tmp;
    }
}
