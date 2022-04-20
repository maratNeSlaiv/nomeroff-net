const p = require('../package');
module.exports = {
    app: {
        name: p.name,
        description: p.description,
        version: p.version
    },
    server: {
        port: process.env.NODE_APP_INSTANCE || 5005
    },
    dataset: {
        baseDir: `${__dir_name}/../../../dataset/ocr/kz`,
        ann: { dir: 'ann', ext: 'json'},
        img: { dir: 'img', ext: 'png'}
    },
    via: {
        partDirs: ['train', 'val']
    },
    moderation: {
        regionOCRModeration: {
            base_dir: "/var/www/nomeroff-net/datasets/options/option_zero",
            options: {
                region_id: [
                    "xx-unknown",
                    "eu-ua-2015",
                    "eu-ua-2004",
                    "eu-ua-1995",
                    "eu",
                    "xx-transit",
                    "ru",
                    "kz",
                    "eu-ua-ordlo-dpr",
                    "eu-ua-ordlo-lpr",
                    "ge",
                    "by",
                    "su",
                    "kg",
                    "am",
                    "ua-military",
                    "ru-military"
                ],
                state_id: ["garbage", "filled", "not filled", "empty"],
                count_lines: ["0", "1", "2", "3"]
            }
        },
        template: {
            tags:[],
            objects:[],
            //"name":"",
            //"description":"",
            state_id:  0,
            region_id: 0,
            size:{
                width:  0,
                height: 0
            },
            moderation:{
                isModerated:0,
                moderatedBy:"dimabendera",
                predicted: ""
            }
        }
    },
};