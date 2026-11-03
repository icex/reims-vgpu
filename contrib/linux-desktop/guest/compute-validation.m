// Manual compute correctness/throughput check. Never installed as a startup job.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <QuartzCore/QuartzCore.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>

int main(void) { @autoreleasepool {
    alarm(120);
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    if(!device){puts("{\"error\":\"no Metal device\"}");return 2;}
    if(!MPSSupportsMTLDevice(device)) {
        puts("{\"available\":false,\"reason\":\"MPSSupportsMTLDevice returned false\"}");return 3;
    }
    const NSUInteger n=1024,bytes=n*n*sizeof(float);
    id<MTLBuffer> a=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> c=[device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if(!a||!b||!c)return 4;
    float *ap=a.contents,*bp=b.contents;
    for(NSUInteger i=0;i<n;i++)for(NSUInteger j=0;j<n;j++){
        ap[i*n+j]=((int)((i*3+j*5)%31)-15)/16.0f;
        bp[i*n+j]=((int)((i*7+j*2)%29)-14)/16.0f;
    }
    MPSMatrixDescriptor *descriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:n columns:n rowBytes:n*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix *am=[[MPSMatrix alloc]initWithBuffer:a descriptor:descriptor];
    MPSMatrix *bm=[[MPSMatrix alloc]initWithBuffer:b descriptor:descriptor];
    MPSMatrix *cm=[[MPSMatrix alloc]initWithBuffer:c descriptor:descriptor];
    MPSMatrixMultiplication *multiply=[[MPSMatrixMultiplication alloc]initWithDevice:device transposeLeft:NO transposeRight:NO resultRows:n resultColumns:n interiorColumns:n alpha:1 beta:0];
    id<MTLCommandQueue> queue=[device newCommandQueue];
    NSMutableArray *rates=[NSMutableArray array];
    unsigned errors=0;double maxError=0;
    for(unsigned batch=0;batch<7;batch++){
        double start=CACurrentMediaTime();
        id<MTLCommandBuffer> command=[queue commandBuffer];
        [multiply encodeToCommandBuffer:command leftMatrix:am rightMatrix:bm resultMatrix:cm];
        [command commit];[command waitUntilCompleted];
        double elapsed=CACurrentMediaTime()-start;
        if(command.status!=MTLCommandBufferStatusCompleted){fprintf(stderr,"MPS command failed: %s\n",command.error.description.UTF8String);return 5;}
        if(batch>=2)[rates addObject:@(2.0*n*n*n/elapsed/1e9)];
    }
    float *cp=c.contents;
    for(unsigned k=0;k<256;k++){
        NSUInteger i=(k*613+7)%n,j=(k*379+11)%n;
        double expected=0;for(NSUInteger t=0;t<n;t++)expected+=(double)ap[i*n+t]*bp[t*n+j];
        double delta=fabs(cp[i*n+j]-expected);maxError=fmax(maxError,delta);
        if(!isfinite(cp[i*n+j])||delta>0.001)errors++;
    }
    NSUInteger nonfinite=0;for(NSUInteger i=0;i<n*n;i++)if(!isfinite(cp[i]))nonfinite++;
    NSArray *sorted=[rates sortedArrayUsingSelector:@selector(compare:)];
    NSDictionary *result=@{@"device":device.name,@"available":@YES,@"passed":@(errors==0&&nonfinite==0),
        @"workload":@"MPS FP32 1024x1024 matrix multiplication",@"cpu_reference_checks":@256,
        @"mismatches":@(errors),@"nonfinite_values":@(nonfinite),@"max_abs_error":@(maxError),
        @"wall_gflops_samples":rates,@"wall_gflops_median":sorted[sorted.count/2]};
    NSData *json=[NSJSONSerialization dataWithJSONObject:result options:0 error:NULL];
    fwrite(json.bytes,1,json.length,stdout);puts("");alarm(0);return errors||nonfinite?1:0;
} }
