package util;

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
        //partition后，在等于区的左边和右边进行递归操作。0
        process(arr,L,a[0]-1);
        process(arr,a[1]+1,R);

    }
    private  int[] partition(int[] arr,int L,int R){
        //定义小于区和大于区
        int less = L-1;
        int most = R;
        int num = arr[R];
        //遍历位置i从最左端开始
        int i = L;
        //每个数的三种情况，
        // 小于num则与把小于区右边的第一个数和当前数交换，i++
        // 大于num则和大于区左边的数交换，
        // 等于则直接跳到下一个
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
        //最后把num与大于区的第一个数交换
        swap(arr,R,most);
        return new int[]{less+1,most};
    }

    public static void swap(int[] arr,int a,int b){
        int tmp = arr[a];
        arr[a] = arr[b];
        arr[b] = tmp;
    }

    /**
     *
     * 归并排序
     */
    public void mergeSort(int[] arr){
        process(arr,0,arr.length-1);
    }

    private void process1(int[] arr,int L,int R){
        if(L==R){
            return;
        }
        int mid = L+((R-L)>>1);
        process(arr,L,mid);
        process(arr,mid+1,R);
        merge(arr,L,mid,R);
    }

    private void merge(int[] arr,int L,int m,int R){
        int[] help = new int[R-L+1];
        int i = 0;
        int p1 = L;
        int p2 = m+1;
        while(p1<=m&&p2<=R){
            if(arr[p1]<=arr[p2]){
                help[i++]=arr[p1++];
            }else {
                help[i++]=arr[p2++];
            }
        }
        while(p1<=m){
            help[i++] = arr[p1++];
        }
        while(p2<=R){
            help[i++] = arr[p2++];
        }
        for(i=0;i<help.length;i++){
            arr[L+i] = help[i];
        }
    }
}
