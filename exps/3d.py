import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource

from exps.smoothed import *
from exps.text_acc import *
from models.base_model import build_model
from settings.test_setting import TestParser

if __name__ == '__main__':
    plt.style.use('default')
    large = 22
    med = 16
    small = 12
    params = {'axes.titlesize': med,
              'legend.fontsize': med,
              'figure.figsize': (10, 10),
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,

              'figure.titlesize': med}
    plt.rcParams.update(params)

    batch_size = 64
    argsv = ['--dataset', 'cifar10', '--exp_id', 'noise_010', '--model_type', 'mini', '--net', 'vgg16', '--data_size',
             '352']
    args = TestParser(argsv).get_args()
    model = build_model(args).cuda()
    ckpt = torch.load(os.path.join(args.model_dir, 'ckpt_best.pth'))
    model.load_weights(ckpt['model_state_dict'])
    model.eval()

    _, test_dataset = set_data_set(args)
    x = test_dataset[0][0]
    batch = (test_dataset[0][0], test_dataset[1][0], test_dataset[3][0])
    # signal = torch.sign(torch.randn_like(batch).to(x.device))
    # n = signal / signal.view(3, -1).norm(p=2, dim=1).view(3, 1, 1, 1) *
    # signal[0] = 0
    # batch = batch + n

    dx = (batch[1] - batch[0]) / 25
    dy = (batch[2] - batch[0]) / 25
    inv_cor = [batch[0] + i * dx + j * dy for i in range(25) for j in range(25)]

    # flat = batch.view(3, -1).numpy()
    # pca = PCA(n_components=2, svd_solver='full')
    # pca.fit(flat)
    #
    # inv = pca.transform(flat)
    # linear = LinearRegression().fit([[-1, 0], [1, 0], [0, 1]], inv)
    # x, y = np.linspace(0, 1, 50), np.linspace(0, 1, 50)
    # xx, yy = np.meshgrid(x, y)
    # inv_cor = [[xx[i, j], yy[i, j]] for i in range(50) for j in range(50)]
    # points = linear.predict(inv_cor)
    x, y = np.linspace(0, 1, 25), np.linspace(0, 1, 25)
    x, y = np.meshgrid(x, y)
    # flat_rev = pca.inverse_transform(points)
    # flat_rev = torch.tensor(flat_rev, dtype=torch.float).view(2500, 3, 32, 32)
    res = []
    for batch_sample in [inv_cor[i:i + 100] for i in range(0, 625, 100)]:
        batch_sample = torch.stack(batch_sample)
        t = model(batch_sample.cuda()).detach().cpu()
        res.append(t)
    res2 = torch.concat(res)

    z = res2[:, 0].reshape(25, 25)
    z = z.numpy()

    # ls = LightSource(270, 45)
    # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ax.view_init(45, 240)
    # plt.show()

    dx2 = 0.054
    dy2 = 0.130

    z2 = np.ones((25, 25)) * z[0][0] + 0.5
    for i in range(25):
        for j in range(25):
            z2[i, j] += dx2 * i + dy2 * j

    ls = LightSource(270, 45)
    rgb = ls.shade(z2, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x, y, z2, rstride=1, cstride=1, facecolors=rgb,
                    linewidth=0, antialiased=False, shade=False)
    ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='red')
    ax.view_init(45, 240)
    ax.set_zlim(-2,7)
    # sns.set_theme(style="default")
    plt.savefig('a', bbox_inches='tight')
    plt.show()

    #
    z3 = z - z2

    ls = LightSource(270, 45)
    rgb = ls.shade(z3, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x, y, z3, rstride=1, cstride=1, facecolors=rgb,
                    linewidth=0, antialiased=False, shade=False)
    ax.view_init(45, 240)
    ax.set_zlim(-2, 7)
    plt.savefig('b', bbox_inches='tight')
    print(1)

    z4 = z2 + z3 * 0.5

    ls = LightSource(270, 45)
    rgb = ls.shade(z4, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    rgb = ls.shade(z2, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    ax.plot_wireframe(x, y, z4, rstride=1, cstride=1, color='red')
    ax.plot_surface(x, y, z2, rstride=1, cstride=1, facecolors=rgb,
                    linewidth=0, antialiased=False, shade=False)
    ax.view_init(45, 240)
    ax.set_zlim(-2, 7)
    plt.savefig('c', bbox_inches='tight')
    print(1)
