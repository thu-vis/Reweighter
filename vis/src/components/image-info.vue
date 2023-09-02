<template>
  <g :id='id' class="fade-in" :transform="`translate(${x},${y})`">
    <!-- <rect :class="`image-rect image-rect-${is_val?'row':'col'}-${idx}`" :idx="idx" :label="sample.label" :width='width' :height='height' :stroke-width="width/4" :stroke="color_map_list[sample.label]" :stroke-opacity="1" ></rect> -->
    <rect :class="`image-rect image-rect-${is_val?'row':'col'}-${idx}`" :idx="idx" :label="sample.label" :width='width' :height='height' :stroke-width="1" stroke="black" :stroke-opacity="1" fill='white'></rect>
    <image :xlink:href='sample.href' :width='imgSize' :height='imgSize'  @contextmenu.prevent="onContextmenu">
    <!-- <rect v-if=is_val :class="`image-rect image-rect-row-${idx}`" :idx="idx" :label="sample.label" :width='width' :height='height' :stroke-width="width/4" :stroke="color_map_list[sample.label]" :stroke-opacity="1" ></rect>
    <circle v-if=!is_val :class="`image-rect image-rect-col-${idx}`" :idx="idx" :label="sample.label" :r='width/2' :cx='width/2' :cy='width/2' :stroke-width="width/4" :stroke="color_map_list[sample.label]" :stroke-opacity="1" ></circle> -->
    
    <!-- <rect :class="`image-rect image-rect-${is_val?'row':'col'}-${idx}`" :idx="idx" :label="sample.label" :width='width+width/3' :height='height+width/3' :x='x-width/6' :y='y-width/6' :stroke-width="1" :stroke="`white`" :stroke-opacity="1" :fill="`none`"></rect> -->
    <!-- <image :xlink:href='sample.href' :width='imgSize' :height='imgSize'  @contextmenu.prevent="onContextmenu" :clip-path="is_val ? '' : 'url(#bipartiteClipCircle)'"> -->
    <title>{{`${idx},${sample.gt_str},${sample.label_str}`}}</title>

    </image>
    <!--<path v-if="sample.corrected" :fill="color_map_list[sample.gt]" :d="STAR" :transform="`translate(${imgSize},${imgSize/6})`"></path>-->
    <rect v-if="showgt && sample.label!=sample.gt" class="wrong"  :width='width/6' :height='height/6' :y="height-height/6" fill="red"></rect>
  </g>
</template>

<script>
  import Vue from 'vue'
  import { mapState, mapActions } from 'vuex'
  import { Shape, Color } from '@/plugins/utils.js';
  import { select, selectAll } from 'd3-selection';
  export default Vue.extend({
    name: 'ImageInfo',
    props: [
        'parentId',
        'idx',
        'x',
        'y',
        'width',
        'height',
        'is_val',
        //'strokeWidth',
        'solid',
    ],
    mounted() {
    },
    computed: {
      ...mapState(['all_samples', 'label_names', 'color_map_list', 'showgt', 'label_names']),
      STAR() {
        return Shape.STAR;
      },
      imgSize() {
        return Math.min(this.width, this.height);
      },
      id() {
        return `${this.parentId}-ImageInfo-${this.idx}`;
      },
      sample() {
        const tmp = this.all_samples[this.idx];
        return {
            // href: "data:image/png;base64," + tmp["image"],
            href: tmp.compressed_image,
            gt: tmp.gt,
            gt_str: this.label_names[tmp.gt],
            label: tmp.label,
            label_str: this.label_names[tmp.label],
            pred: tmp.pred,
            pred_str: this.label_names[tmp.pred],
        }
      }
    },
    methods: {
      ...mapActions(['fetch_relationship']),
      // onContextmenu(event) {
      //   this.$contextmenu({
      //     items: this.label_names.map((name, l) => ({label:name, onClick:() => {
      //       console.log('click', this.idx, name);
      //       let sample = this.all_samples[this.idx];
      //       console.assert(sample.gt==l, sample);
      //       sample.corrected = true;
      //       if (this.is_val) {
      //         sample.set_lam = 0;
      //       } else {
      //         if (sample.gt != sample.label) {
      //             sample.is_pos = false;
      //             sample.is_neg = true;
      //             sample.shape = Shape.DOWN_TRIANGLE;
      //             sample.large_shape = Shape.LARGE_DOWN_TRIANGLE;
      //             sample.color = Color.RED_POINT;
      //         } else {
      //             sample.is_pos = true;
      //             sample.is_neg = false;
      //             sample.shape = Shape.UP_TRIANGLE;
      //             sample.large_shape = Shape.LARGE_UP_TRIANGLE;
      //             sample.color = Color.GREEN_POINT;
      //         }
      //         this.$forceUpdate();
      //         this.$parent.$forceUpdate();
      //       }
      //     }})),
      //     event,
      //     customClass: "custom-class",
      //     zIndex: 3,
      //     minWidth: 130
      //   });
      //   return false;
      // }
      onMarkClean() {
        const sample = this.all_samples[this.idx];
        sample.is_pos = true;
        sample.is_neg = false;
        sample.color = Color.GREEN_POINT;
      },
      onMarkNoisy() {
        const sample = this.all_samples[this.idx];
        sample.is_pos = false;
        sample.is_neg = true;
        sample.color = Color.RED_POINT;
        // this.subcluster()
      },
      async onAddReward() {
        await this.fetch_relationship({'row_index': [this.idx]});
      },
      onChangeLabel(label) {
        console.log(`change label of ${this.idx} from ${this.all_samples[this.idx].label} to ${label}`)
      },
      onRemove() {
        console.log('remove validation', this.idx)
      },
      onIncreaseWeight() {
        console.log('increase weight', this.idx)
      },
      onDecreaseWeight() {
        console.log('decrease weight', this.idx)
      },
      onContextmenu(event) {
        if (this.is_val) {
          this.$contextmenu({
            items: [
          {
            label: "Relabel",
            icon: "icon-relabel",
            children: this.label_names.map((name, l)=>({
                label: name,
                onClick: () => {
                  this.onChangeLabel(l)
                },
                icon: l == this.all_samples[this.idx].label ? "icon-check" : "",
              })
            ),
          },
          {
            label: "Remove",
            icon: "icon-remove",
            onClick: this.onRemove,
          },
          {
            label: "Increase weight",
            icon: "icon-good",
            onClick: this.onIncreaseWeight,
          },
          {
            label: "Decrease weight",
            icon: "icon-bad",
            onClick: this.onDecreaseWeight,
          },
        ],
            event,
            customClass: "custom-class",
            zIndex: 3,
            minWidth: 130
          });
          return false;
        } else {
          const options = [['Mark clean', this.onMarkClean, "icon-good"],
                          ['Mark noisy', this.onMarkNoisy, "icon-bad"], 
                          ['Add into validation', this.onAddReward, "icon-add"]];
          this.$contextmenu({
            items: options.map((option, l) => ({label:option[0], onClick:option[1], icon: option[2]})),
            event,
            customClass: "custom-class",
            zIndex: 3,
            minWidth: 130
          });
          return false;
        }
      }
    }
});
</script>


<style>

/*
.fade-in {
  animation-name: fadeIn;
  animation-duration: 1s;
}
/*
.fade-out {
  animation-name: fadeOut;
  animation-duration: 1s;
}
*/



</style>
