#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreVideo/CoreVideo.h>
#include <math.h>

static NSDictionary *modeInfo(CGDisplayModeRef mode) {
    return @{@"id":@(CGDisplayModeGetIODisplayModeID(mode)),
        @"width":@(CGDisplayModeGetWidth(mode)), @"height":@(CGDisplayModeGetHeight(mode)),
        @"pixel_width":@(CGDisplayModeGetPixelWidth(mode)), @"pixel_height":@(CGDisplayModeGetPixelHeight(mode)),
        @"refresh_hz":@(CGDisplayModeGetRefreshRate(mode)), @"io_flags":@(CGDisplayModeGetIOFlags(mode))};
}

int main(int argc, const char **argv) { @autoreleasepool {
    CGDirectDisplayID display=CGMainDisplayID();
    if(!display){fputs("{\"error\":\"No GUI display; log in first\"}\n",stdout);return 3;}
    NSArray *modes=CFBridgingRelease(CGDisplayCopyAllDisplayModes(display,
        (__bridge CFDictionaryRef)@{(__bridge id)kCGDisplayShowDuplicateLowResolutionModes:@YES}));
    NSMutableArray *values=[NSMutableArray array];
    CGError changed=kCGErrorSuccess;BOOL selected=argc==1;
    for(id item in modes) {
        CGDisplayModeRef mode=(__bridge CGDisplayModeRef)item;
        NSDictionary *info=modeInfo(mode);[values addObject:info];
        if((argc==5||argc==6) && CGDisplayModeGetWidth(mode)==strtoul(argv[1],NULL,10) &&
           CGDisplayModeGetHeight(mode)==strtoul(argv[2],NULL,10) &&
           CGDisplayModeGetPixelWidth(mode)==strtoul(argv[3],NULL,10) &&
           CGDisplayModeGetPixelHeight(mode)==strtoul(argv[4],NULL,10) &&
           (argc==5||fabs(CGDisplayModeGetRefreshRate(mode)-strtod(argv[5],NULL))<0.1)) {
            CGDisplayConfigRef config=NULL;
            changed=CGBeginDisplayConfiguration(&config);
            if(!changed)changed=CGConfigureDisplayWithDisplayMode(config,display,mode,NULL);
            if(!changed)changed=CGCompleteDisplayConfiguration(config,kCGConfigurePermanently);
            else if(config)CGCancelDisplayConfiguration(config);
            selected=YES;break;
        }
    }
    CGDisplayModeRef current=CGDisplayCopyDisplayMode(display);
    CVDisplayLinkRef link=NULL;CVReturn linkStatus=CVDisplayLinkCreateWithCGDisplay(display,&link);
    CVTime nominal=link?CVDisplayLinkGetNominalOutputVideoRefreshPeriod(link):(CVTime){0};
    double hz=nominal.timeValue?(double)nominal.timeScale/nominal.timeValue:0;
    NSMutableArray *screens=[NSMutableArray array];
    for(NSScreen *screen in NSScreen.screens) [screens addObject:@{@"width":@(screen.frame.size.width),
        @"height":@(screen.frame.size.height),@"backing_scale":@(screen.backingScaleFactor)}];
    NSDictionary *result=@{@"display_id":@(display),@"current":modeInfo(current),@"modes":values,
        @"screens":screens,@"selected":@(selected),@"set_status":@(changed),
        @"display_link_status":@(linkStatus),@"display_link_nominal_hz":@(hz)};
    NSData *json=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:NULL];
    fwrite(json.bytes,1,json.length,stdout);puts("");
    if(current)CGDisplayModeRelease(current);if(link)CVDisplayLinkRelease(link);
    return selected&&changed==kCGErrorSuccess?0:1;
} }
