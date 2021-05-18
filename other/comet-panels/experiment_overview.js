class MyPanel extends Comet.Panel {
  setup() {
    this.options = {};
  }
  
  find_par(parameters, name) {
    const par_name = parameters.find(element => element.name == name);
    if (typeof(par_name) != "undefined") {
      return par_name.valueMax;
    }
    return "-";
  }
  
  find_metric_min(metrics, name) {
    const metric = metrics.find(element => element.name == name);
    if (typeof(metric) != "undefined") {
      return Number(metric.valueMin).toFixed(4);
    }
    return 0;
  }
  
  find_metric_max(metrics, name) {
    const metric = metrics.find(element => element.name == name);
    if (typeof(metric) != "undefined") {
      return Number(metric.valueMax).toFixed(4);
    }
    return 0;
  }
    
  async draw(experimentKeys) {
    const promises = experimentKeys.map(async experimentKey => {
		const result = {};
      result.metadata = await this.api.experimentMetadata(experimentKey);
      result.parameters = await this.api.experimentParameters(experimentKey);
      result.metrics = await this.api.experimentMetricsSummary(experimentKey);
      
      //this.print(".", false);
      return result;
    });
   
    const results = await Promise.all(promises);
    //console.log(results);
    const data = results.map(result => {
      const url = $("<a>")
        .attr("href", this.api.experimentURL(result.experimentKey))
        .attr("target", "_top")
        .text(result.metadata.experimentName)
        .get(0);

      return { 
        exp: url, 
        val_loss: this.find_metric_min(result.metrics, "val_loss"),
        val_acc: this.find_metric_max(result.metrics, "val_acc"),
        val_patch_acc: this.find_metric_max(result.metrics, "val_patch_acc"),
        model: this.find_par(result.parameters, "Model"),
        train_collections: this.find_par(result.parameters, "Training Collections"),
        transformations: this.find_par(result.parameters, "Transformations").split(",").length,
        img_size: this.find_par(result.parameters, "Img size"),
        frg_thrsh: this.find_par(result.parameters, "Foreground Threshold"),
        loss_fun: this.find_par(result.parameters, "Loss-Function"),
        optim: this.find_par(result.parameters, "Optimizer"),
        optim_lr: this.find_par(result.parameters, "Optimizer-LR"),
        lr_sched: this.find_par(result.parameters, "lr-sched."),
      };
    });

    this.print("<h5>Summary of all experiments</h5>");
	//console.log(data);
    new Table(
      data,
      ["exp", "val_loss", "val_acc", "val_patch_acc", "model", "train_collections", "transformations", "img_size", "frg_thrsh", "optim", "optim_lr", "lr_sched", "loss_fun" ],
      { exp: "Experiment Id", 
       val_loss: "Val loss", 
       val_acc: "Val acc",  
       val_patch_acc: " Val patch acc", 
       model: "Model",
       train_collections: "Training Collections",
       transformations: "Nr. Trans",
       img_size: "Img size",
       frg_thrsh: "Frg Thresh.",
       loss_fun : "Loss function",
       optim: "Optim",
       optim_lr: "Optim. LR",
       lr_sched: "Lr Sched." },
      "exp-time-table",
      "table table-striped"
    ).appendTo(this.id);
  }
}