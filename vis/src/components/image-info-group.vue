<template>
<g v-if="group.nrow>=1">
 <g :id='id' class="fade-in" :transform="`translate(${group.x*imgSize},${4})`">
   <text :x='imgSize*(group.cnt+(group.cnt-1)*group.margin)/2' :y=-3 text-anchor="middle" font-size="0.7em">{{group.label_name}}</text>
   <ImageInfo v-for="(idx, i) in display_indices" :key='`Image-${id}-${i}-${idx}`' :parentId='id' :idx="idx" :width='imgSize' :height='imgSize' :x='(imgSize)*(1+group.margin)*(i%group.cnt)' :y='(imgSize)*(1+group.margin) * Math.floor(i / group.cnt)' :is_val="group.is_val"></ImageInfo>
  </g>
</g>
<g v-else>
  <g :id='id' class="fade-in" :transform="`translate(${group.x*imgSize},${4})`">
  <text :x='imgSize*(group.cnt+(group.cnt-1)*group.margin)/2' :y=2 text-anchor="middle" font-size="0.7em">{{group.label_name}}</text>
  </g>
</g>
</template>

<script>
  import Vue from 'vue'
  import { mapState, mapActions } from 'vuex'
  import { Shape, Color } from '@/plugins/utils.js';
  import { select, selectAll } from 'd3-selection';
  export default Vue.extend({
    name: 'ImageInfoGroup',
    components: {
        ImageInfo: () => import('./image-info.vue'),
    },
    props: [
        'parentId',
        'group',
        'is_val',
        //'strokeWidth',
        'solid',
        'imgSize',
    ],
    mounted() {
      // console.log(this.group.indices, this.group.cnt * this.group.nrow, this.group.cnt, this.group.nrow, this.group.indices.slice(0, this.group.cnt * this.group.nrow))
    },
    computed: {
      ...mapState(['all_samples', 'label_names', 'color_map_list', 'showgt', 'label_names']),
      id() {
        return `${this.parentId}-ImageInfoGroup-${this.group.label_name}`;
      },
      display_indices() {
        return this.group.indices.slice(0, this.group.cnt *  this.group.nrow)
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
