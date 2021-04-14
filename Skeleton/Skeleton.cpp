//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Bertok Attila
// Neptun : I7XH6P
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

enum MaterialType {ROUGH, REFLECTIVE};
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	bool portalMaterial;
	vec3 pentagonCenter;
	Material(MaterialType t) {
		type = t;
		pentagonCenter = vec3(0, 0, 0);
	}
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks; 
		shininess = _shininess;
		portalMaterial = false;
	}
};
vec3 operator/(vec3 num, vec3 denum) {
	return vec3(num.x / denum.x, num.y / denum.y, num.z / denum.z);
}
struct ReflectiveMaterial : Material
{
	ReflectiveMaterial(vec3 n, vec3 kappa, bool isPolral) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
		portalMaterial = isPolral;
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct GoldObject : public Intersectable {
	float aParam, bParam, cParam;
	GoldObject(float aParam, float bParam, float cParam) {
		this->aParam = aParam;
		this->bParam = bParam;
		this->cParam = cParam;
		vec3 n(0.17, 0.35, 1.5); vec3 kappa(3.1, 2.7, 1.9);
		material = new ReflectiveMaterial(n, kappa, false);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		float a = aParam * pow(ray.dir.x, 2) + bParam * pow(ray.dir.y, 2);
		float b = 2.0f * aParam * ray.start.x * ray.dir.x + 2.0f * bParam * ray.start.y * ray.dir.y - cParam * ray.dir.z;
		float c = aParam * pow(ray.start.x, 2) + bParam * pow(ray.start.y, 2) - cParam * ray.start.z;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		if (insideSphere(ray.start + ray.dir * t2)) {
			hit.t = t2;
		}
		else if(insideSphere(ray.start + ray.dir * t1)){
			hit.t = t1;
		}
		else {
			return hit;
		}
		hit.position = ray.start + ray.dir * hit.t;

		float tmp = exp(aParam * pow(hit.position.x, 2) + bParam * pow(hit.position.y, 2) - cParam * hit.position.z);
		vec3 gradient = vec3( tmp * 2 * hit.position.x * aParam, tmp * 2 * hit.position.y * bParam, -cParam * tmp);
		hit.normal = -normalize(gradient);
		hit.material = material;
		return hit;
	}
	bool insideSphere(vec3 point) {
		if (sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2)) < 0.3f) return true;
		return false;
	}
};

struct Face {
	vec3 points[5];
	vec3 normal() {
		return -normalize(cross(points[1] - points[0], points[2]-points[0]));
	}
};

struct Dodecahedron : Intersectable{
	Face faces[12];
	Material* material2;
	Dodecahedron() {
		vec3 kd(0.73f, 0.82f, 0.65f), ks(2, 2, 2);
		material = new RoughMaterial(kd, ks, 100);

		vec3 n(1, 1, 1); vec3 kappa(10, 10, 10);
		material2 = new ReflectiveMaterial(n, kappa, true);

		std::vector<vec3> objVertices;
		objVertices.push_back(vec3(0, 0.618, 1.618));	 objVertices.push_back(vec3(0, -0.618, 1.618));
		objVertices.push_back(vec3(0, -0.618, -1.618));	 objVertices.push_back(vec3(0, 0.618, -1.618));
		objVertices.push_back(vec3(1.618, 0, 0.618));	 objVertices.push_back(vec3(-1.618, 0, 0.618));
		objVertices.push_back(vec3(-1.618, 0, -0.618));	 objVertices.push_back(vec3(1.618, 0, -0.618));
		objVertices.push_back(vec3(0.618, 1.618, 0));	 objVertices.push_back(vec3(-0.618, 1.618, 0));
		objVertices.push_back(vec3(-0.618, -1.618, 0));  objVertices.push_back(vec3(0.618, -1.618, 0));
		objVertices.push_back(vec3(1, 1, 1));			 objVertices.push_back(vec3(-1, 1, 1));
		objVertices.push_back(vec3(-1, -1, 1));			 objVertices.push_back(vec3(1, -1, 1));
		objVertices.push_back(vec3(1, -1, -1));			 objVertices.push_back(vec3(1, 1, -1));
		objVertices.push_back(vec3(-1, 1, -1));			 objVertices.push_back(vec3(-1, -1, -1));
		
		int indexes[12][5] = {	{1, 2, 16, 5, 13}, 
								{1, 13, 9, 10, 14},
								{1, 14, 6, 15, 2},
								{2, 15, 11, 12, 16},
								{3, 4, 18, 8, 17},
								{3, 17, 12, 11, 20},
								{3, 20, 7, 19, 4},
								{19, 10, 9, 18, 4},
								{16, 12, 17, 8, 5},
								{5, 8, 18, 9, 13},
								{14, 10, 19, 7, 6},
								{6, 7, 20, 11, 15} };

		for (int i = 0; i < 12; i++)
		{
			Face tmpFace;
			for (int j = 0; j < 5; j++)
			{
				tmpFace.points[j] = objVertices.at(indexes[i][j] - 1);
			}
			faces[i] = tmpFace;
		}
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		int bestFaceIndex = 0;
		float smallestPositiveT = -1;
		for (int i = 0; i < 12; i++)
		{
			float nx = faces[i].normal().x;
			float ny = faces[i].normal().y;
			float nz = faces[i].normal().z;
			float x0 = faces[i].points[0].x;
			float y0 = faces[i].points[0].y;
			float z0 = faces[i].points[0].z;
			float t = (nx * x0 + ny * y0 + nz * z0 - nx * ray.start.x - ny * ray.start.y - nz * ray.start.z) / (nx * ray.dir.x + ny * ray.dir.y + nz * ray.dir.z); 
			if (t > 0 && (smallestPositiveT < 0 || smallestPositiveT > t)) {
				smallestPositiveT = t;
				bestFaceIndex = i;
			}
		}
		if (smallestPositiveT > 0) {
			hit.t = smallestPositiveT;
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = faces[bestFaceIndex].normal();
			hit.material = material;
			if (closestSideDistance(bestFaceIndex, hit.position) > 0.1) {
				vec3 pentagonCenter = vec3(0, 0, 0);
				for (int i = 0; i < 5; i++)
				{
					pentagonCenter = pentagonCenter + faces[bestFaceIndex].points[i];
				}
				pentagonCenter = pentagonCenter / 5.0f;
				hit.material = material2;
				hit.material->pentagonCenter = pentagonCenter;
			}
		}
		return hit;
	}
	float distanceFromLine(vec3 point, vec3 linePoint1, vec3 linePoint2) {
		return length(cross((linePoint2 - linePoint1), (linePoint1 - point))) / length(linePoint2 - linePoint1);
	}
	float closestSideDistance(int faceIndex, vec3 point) {
		float dist = -1;
		for (int i = 0; i < 4; i++)
		{
			float tmpDist = distanceFromLine(point, faces[faceIndex].points[i], faces[faceIndex].points[i + 1]);
			if (tmpDist < dist || dist < 0) dist = tmpDist;
		}
		float tmpDist = distanceFromLine(point, faces[faceIndex].points[0], faces[faceIndex].points[4]);
		if (tmpDist < dist || dist < 0) dist = tmpDist;
		return dist;
	}
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		this->fov = fov;
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye, lookat, up, fov);
	}
};

struct PointLight {
	vec3 location;
	vec3 Le;
	PointLight(vec3 _location, vec3 _Le) {
		this->location = _location;
		Le = _Le;
	}
};

const float epsilon = 0.01f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<PointLight*> pointLights;
	Camera camera;
	vec3 La;
public:
	void animate(float dt) {
		camera.Animate(dt);
	}
	void build() {
		vec3 eye = vec3(0, 0, 1.3), vup = vec3(1, 1, 1), lookat = vec3(0, 0, 0);
		float fov = 55 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		La = vec3(0.1f, 0.1f, 0.1f);

		vec3 position(0, 0.6, 0), LePoint(1, 1, 1);
		pointLights.push_back(new PointLight(position, LePoint));

		objects.push_back(new Dodecahedron());
		objects.push_back(new GoldObject(12.0f, 12.0f, 1.0f));
	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
	bool shadowIntersectPointLight(Ray ray, float dist) {
		Hit shadowHit = firstIntersect(ray);
		if ((shadowHit.t < 0 || shadowHit.t > dist)) return true;
		return false;
	}

	vec3 rotatePoint(vec3 pointToRotate, vec3 rotationAxisNormal, vec3 pentagonCenter) {
		pointToRotate = pointToRotate - pentagonCenter;
		float sinTheta = sin(0.4 * M_PI);
		float cosTheta = cos(0.4 * M_PI);
		// forras: sajat a kod, de az egyszerusege miatt a Rodrigues' rotation formulat implementaltam az eloadason tanult kvaternios megoldas helyett, aminek a mukodesenek tobb oldalon olvastam utana, pl.: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
		vec3 rotated = (pointToRotate * cosTheta) + (cross(rotationAxisNormal, pointToRotate) * sinTheta) + (rotationAxisNormal * dot(rotationAxisNormal, pointToRotate)) * (1 - cosTheta);
		rotated = rotated + pentagonCenter;
		return rotated;
	}
	vec3 rotateVector(vec3 pointToRotate, vec3 rotationAxisNormal) {
		float sinTheta = sin(0.4 * M_PI);
		float cosTheta = cos(0.4 * M_PI);
		vec3 rotated = (pointToRotate * cosTheta) + (cross(rotationAxisNormal, pointToRotate) * sinTheta) + (rotationAxisNormal * dot(rotationAxisNormal, pointToRotate)) * (1 - cosTheta);
		return rotated;
	}
	vec3 trace(Ray ray, int depth = 0, int portalDepth = 0) {
		if (depth > 5) return La;
		if (portalDepth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		if (hit.material->type == ROUGH) {
				for (PointLight* pointLight : pointLights) { 
					vec3 lightVector = normalize( pointLight->location - hit.position); 
					float distanceFromPointLight = length(pointLight->location - hit.position); 
					Ray shadowRay(hit.position + hit.normal * epsilon, lightVector);
					float cosTheta = dot(hit.normal, lightVector);
					if (cosTheta > 0 && shadowIntersectPointLight(shadowRay, distanceFromPointLight)) {	
						vec3 Le = pointLight->Le * (1.0f / pow(distanceFromPointLight, 2));
						outRadiance = outRadiance + Le * hit.material->kd * cosTheta; 
						vec3 halfway = normalize(-ray.dir + lightVector); 
						float cosDelta = dot(hit.normal, halfway);
						if (cosDelta > 0) outRadiance = outRadiance + Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
					}
				}
		}
		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			if (hit.material->portalMaterial) {
				vec3 newStartPoint = rotatePoint(hit.position, hit.normal, hit.material->pentagonCenter);
				vec3 newDirection = rotateVector(reflectedDir, hit.normal);
				outRadiance = outRadiance + F * trace(Ray(newStartPoint + hit.normal * epsilon, newDirection), depth, portalDepth + 1);
			}
			else {
				outRadiance = outRadiance + F * trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1, portalDepth);
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram; 
Scene scene;

const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;
	
	int windowWidth, windowHeight;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight)
	{
		this->windowWidth = windowWidth;
		this->windowHeight = windowHeight;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);		
		unsigned int vbo;		
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	    
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);   
	}

	void Draw(std::vector<vec4>& image){
		Texture texture(windowWidth, windowHeight, image);
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->Draw(image);
	glutSwapBuffers();							
}

void onKeyboard(unsigned char key, int pX, int pY) {}
void onKeyboardUp(unsigned char key, int pX, int pY) {}
void onMouse(int button, int state, int pX, int pY) {}
void onMouseMotion(int pX, int pY) {}
void onIdle() {
	scene.animate(0.1f);
	glutPostRedisplay();
}